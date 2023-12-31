import os
import subprocess
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import audio
import face_detection
from models import Wav2Lip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for inference.")


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i: i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


print("Loading face detector")
detector = face_detection.FaceAlignment(flip_input=False, device=device)
print("Face detector loaded")


def face_detect(images):
	batch_size = 16

	while True:
		predictions = []
		try:
			for i in range(0, len(images), batch_size):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i: i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError("Image too big to run face detection on GPU.")
			batch_size //= 2
			print("Recovering from OOM error; New batch size: {}".format(batch_size))
			continue
		break

	results = []
	# top, bottom, left, right
	pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
	for rect, image in zip(predictions, images):
		if rect is None:
			raise ValueError("Face not detected! Ensure the video contains a face in all the frames.")

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

	return results


def datagen(frames, mels, batch_size):
	face_size = 96

	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	print("Run face detection")
	face_det_results = face_detect(frames)
	print("End face detection")

	for i, m in enumerate(mels):
		idx = i % len(frames)
		frame_to_save = frames[idx]
		face, coords = face_det_results[idx]

		face = cv2.resize(face, (face_size, face_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, face_size // 2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, face_size // 2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
		mel_batch = np.reshape(
			mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
		)

		yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size: int = 16
fourcc = 0x30323449  # I420 (yuv420)


def load_model(path) -> Wav2Lip:
	wav2lip = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = torch.load(path, torch.device(device))
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace("module.", "")] = v
	wav2lip.load_state_dict(new_s)
	return wav2lip.to(device).eval()


print("Loading model")
model: Wav2Lip = load_model("checkpoints/wav2lip.pth")
print("Model loaded")


class VideoCache:
	def __init__(self, fps, frames):
		self.fps = fps
		self.frames = frames


video_cache = {}


def main(in_audio: str, in_video: str, out_video: str):
	tmpout = f"/tmp/result-{uuid.uuid4()}.nut"

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

		while True:
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

	print("Start generating mel chunks")
	while True:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
		i += 1
	print("End generating mel chunks")

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[: len(mel_chunks)]

	print("Start datagen")
	batch_size = 128
	gen = datagen(full_frames, mel_chunks, batch_size)
	print("End datagen")

	frame_h, frame_w = full_frames[0].shape[:-1]
	out = cv2.VideoWriter(tmpout, fourcc, fps, (frame_w, frame_h))

	for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
		img_batch = torch.FloatTensor(img_batch).to(device)
		mel_batch = torch.FloatTensor(mel_batch).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy() * 255.0

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = "ffmpeg -y -i {} -i {} -strict -2 -c:v h264_nvenc -preset fast {}".format(in_audio, tmpout, out_video)
	subprocess.call(command, shell=True)
	os.remove(tmpout)


app = FastAPI()


class Request(BaseModel):
	audio_path: str
	video_path: str


@app.post("/", response_class=PlainTextResponse)
async def generate(request: Request):
	out = f"/result/{uuid.uuid4()}.mp4"
	main(request.audio_path, request.video_path, out)
	return out
