import cv2
import time
import torch
import numpy as np
import torchvision.transforms.functional as F
from preprocess.gen_data import gaussian_kernel
from test.test_cpm import get_key_points, draw_image

if __name__ == "__main__":
	model = torch.load('../model/best_cpm.pth').cuda()

	capture = cv2.VideoCapture(0)

	while (True):
		start_time = time.time()
		ret, image = capture.read()
		ori_image = image

		height, width, _ = image.shape
		image = np.asarray(image, dtype=np.float32)
		image = cv2.resize(image, (368, 368), interpolation=cv2.INTER_CUBIC)

		# Normalize
		image -= image.mean()
		image = F.to_tensor(image)

		# Generate center map
		centermap = np.zeros((368, 368, 1), dtype=np.float32)
		kernel = gaussian_kernel(size_h=368, size_w=368, center_x=184, center_y=184, sigma=3)
		kernel[kernel > 1] = 1
		kernel[kernel < 0.01] = 0
		centermap[:, :, 0] = kernel
		centermap = torch.from_numpy(np.transpose(centermap, (2, 0, 1)))

		image = torch.unsqueeze(image, 0).cuda()
		centermap = torch.unsqueeze(centermap, 0).cuda()

		model.eval()
		input_var = torch.autograd.Variable(image)
		center_var = torch.autograd.Variable(centermap)

		heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, center_var)
		key_points = get_key_points(heat6, height=height, width=width)

		image = draw_image(ori_image, key_points)

		# calculate fps
		end_time = time.time()
		second = end_time - start_time
		fps = 1 / second
		print('fps:', fps)

		cv2.imshow('real time', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
