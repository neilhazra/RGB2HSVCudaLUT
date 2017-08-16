#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <algorithm>    // std::min

using namespace std;
using namespace cv;
using namespace cv::gpu;




inline uint getFirstIndex(uchar, uchar, uchar);
uchar *LUMBGR2HSV;
uchar *d_LUMBGR2HSV;

__global__
void kernelconvert(uchar *LUT)
{
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
	uint k = (blockIdx.z * blockDim.z) + threadIdx.z;

	if (i < 256 && j < 256 && k < 256) {
		

		uchar _b = i;
		uchar _g = j;
		uchar _r = k;
		float b = (float)_b / 255.0;
		float g = (float)_g / 255.0;
		float r = (float)_r / 255.0;
		float h, s, v;
		float _min = min(min(b, g), r);
		v = max(max(b, g), r);
		float chroma = v - _min;
		if (v != 0)
			s = chroma / v; // s
		else {
			s = 0;
			h = -1;
			return;
		}
		if (r == v)
			h = (g - b) / chroma;
		else if (g == v)
			h = 2 + (b - r) / chroma;
		else
			h = 4 + (r - g) / chroma;
		h *= 30;
		if (h < 0)	h += 180;
		s *= 255;
		v *= 255;
		uint index = 3 * 256 * 256 * i + 256 * 3 * j + 3 * k;
		LUT[index] = (uchar)h;
		LUT[index + 1] = (uchar)s; //height, width  Saturation
		LUT[index + 2] = (uchar)v; //height, width  Value
	}
}

__global__
void kernelSwap(PtrStepSz<uchar> src, PtrStepSz<uchar>  dst, uchar *LUT) {
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint j = 3 * ((blockIdx.y * blockDim.y) + threadIdx.y);
	uint index = 3 * 256 * 256 * src.ptr(i)[j] + 256 * 3 * src.ptr(i)[j + 1] + 3 * src.ptr(i)[j + 2];

	dst.ptr(i)[j] = LUT[index];
	dst.ptr(i)[j+1] = LUT[index+1];
	dst.ptr(i)[j+2] = LUT[index+2];
}

inline uint getFirstIndex(uchar b, uchar g, uchar r) {
	return 3 * 256 * 256 * b + 256 * 3 * g + 3 * r;
}

void initializeLUM() {
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc((void **)&LUMBGR2HSV, 256*256*256*3, cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&d_LUMBGR2HSV, (void *) LUMBGR2HSV, 0);
	
	dim3 threads_per_block(8, 8,8);
	dim3 numBlocks(32,32,32);
	
	kernelconvert << <numBlocks, threads_per_block >> >(d_LUMBGR2HSV);
	
}

void BGR2HSV_LUM(GpuMat src, GpuMat dst) {
	dim3 threads_per_block(16, 16);
	dim3 numBlocks(45, 80);
	kernelSwap << <numBlocks, threads_per_block >> >(src, dst, d_LUMBGR2HSV);
	
}

/*
Commented Sections used to test speed difference with GPU look up table and opencv cvtColor
GPU code about 2760000 cv ticks faster
*/
int main(int argc, char** argv)
{	
	string filename = "mouse.mp4";
	initializeLUM();
	gpu::setDevice(0);
	gpu::GpuMat  src, inHSV;
	Mat frame;

	Mat openCvcvt;
	VideoCapture capture(filename);

	for (; ; )
	{
		capture.read(frame);
		if (frame.empty())
			break;

		src.upload(frame);
		inHSV.upload(frame);


		//int64 before = getTickCount();
		//cvtColor(frame, openCvcvt, CV_BGR2HSV);
		//int64 afterOpencvF = getTickCount();
		BGR2HSV_LUM(src, inHSV);
		//int64 afterCuda = getTickCount();

		//int cvtColorTime = afterOpencvF - before;
		//int kernelConvertTime = afterCuda - afterOpencvF;

		//printf("CvtColor: %d kernelConvert %d diff %d \n", cvtColorTime, kernelConvertTime, cvtColorTime - kernelConvertTime);
		
		Mat download(inHSV);
		imshow("HSV", download);
		//imshow("hsvopencv", openCvcvt);

		waitKey(10); // waits to display frame
	}
	waitKey(0); // key press to close window
}
