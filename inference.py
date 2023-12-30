from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uuid

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T :]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(
		face_detection.LandmarksType._2D, flip_input=False, device=device
	)

	batch_size = 16

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(
					detector.get_detections_for_batch(
						np.array(images[i : i + batch_size])
					)
				)
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError(
					"Image too big to run face detection on GPU. Please use the --resize_factor argument"
				)
			batch_size //= 2
			print("Recovering from OOM error; New batch size: {}".format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = [0, 10, 0, 0]  # TODO
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite(
				"temp/faulty_frame.jpg", image
			)  # check this frame where the face was not detected.
			raise ValueError(
				"Face not detected! Ensure the video contains a face in all the frames."
			)

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)

		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	boxes = get_smoothened_boxes(boxes, T=5)
	results = [
		[image[y1:y2, x1:x2], (y1, y2, x1, x2)]
		for image, (x1, y1, x2, y2) in zip(images, boxes)
	]

	del detector
	return results

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection

	for i, m in enumerate(mels):
		idx = i % len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (96, 96))  # TODO

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= 128:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, 96 // 2 :] = 0  # TODO 96

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
			mel_batch = np.reshape(
				mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
			)

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, 96 // 2 :] = 0  # TODO 96

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
		mel_batch = np.reshape(
			mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
		)

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} for inference.".format(device))

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = torch.load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace("module.", "")] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

model = load_model("checkpoints/wav2lip.pth")
print("Model loaded")

class VideoCache:
	def __init__(self, fps, frames):
		self.fps = fps
		self.frames = frames

video_cache = {}

def main(in_audio: str, in_video: str, out_video: str):
	fps = 0.0
	full_frames = []

	if in_video in video_cache:
		print("Using cached video")
		val = video_cache[in_video]
		fps = val.fps
		full_frames = val.frames
	else:
		video_stream = cv2.VideoCapture(in_video)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print("Reading video frames...")

		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			full_frames.append(frame)
		video_cache[in_video] = VideoCache(fps, full_frames)

	print("Number of frames available for inference: " + str(len(full_frames)))

	wav = audio.load_wav(in_audio, 16000)
	mel = audio.melspectrogram(wav)

	mel_chunks = []
	mel_idx_multiplier = 80.0 / fps
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[: len(mel_chunks)]

	batch_size = 128
	gen = datagen(full_frames.copy(), mel_chunks)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(
		tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
	):
		if i == 0:
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(
				"temp/result.avi",
				cv2.VideoWriter_fourcc(*"h264"),
				fps,
				(frame_w, frame_h),
			)

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = "ffmpeg -y -i {} -i {} -strict -2 -c:v copy {}".format(
		in_audio, "temp/result.avi", out_video
	)
	subprocess.call(command, shell=platform.system() != "Windows")

app = FastAPI()

class Request(BaseModel):
	audio_path: str
	video_path: str

@app.post("/", response_class=PlainTextResponse)
async def generate(request: Request):
	out = f"/result/{uuid.uuid4()}.mp4"
	main(request.audio_path, request.video_path, out)
	return out
