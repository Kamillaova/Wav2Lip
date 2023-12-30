from __future__ import print_function

from .utils import *


class FaceAlignment:
	def __init__(
		self,
		device="cuda",
		flip_input=False,
		face_detector="sfd",
		verbose=False,
	):
		self.device = device
		self.flip_input = flip_input
		self.verbose = verbose

		if "cuda" in device:
			torch.backends.cudnn.benchmark = True

		# Get the face detector
		face_detector_module = __import__(
			"face_detection.detection." + face_detector,
			globals(),
			locals(),
			[face_detector],
			0,
		)
		self.face_detector = face_detector_module.FaceDetector(
			device=device, verbose=verbose
		)

	def get_detections_for_batch(self, images):
		images = images[..., ::-1]
		detected_faces = self.face_detector.detect_from_batch(images.copy())
		results = []

		for i, d in enumerate(detected_faces):
			if len(d) == 0:
				results.append(None)
				continue
			d = d[0]
			d = np.clip(d, 0, None)

			x1, y1, x2, y2 = map(int, d[:-1])
			results.append((x1, y1, x2, y2))

		return results
