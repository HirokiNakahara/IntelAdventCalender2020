#include<iostream>
#include<fstream>
#include<string>
#include<cstdlib>
#include<cmath>

#include"CL/opencl.h"
#include"AOCLUtils/aocl_utils.h"

using namespace aocl_utils;
using namespace std;

cl_platform_id   platform = NULL;
cl_device_id     device;
cl_context       context = NULL;
cl_command_queue queue;
cl_program       program = NULL;
cl_kernel        kernel;
cl_mem           inp_buf;
cl_mem           out_buf;
cl_event         kevent;

bool init_opencl();
void cleanup();

// testbench from Google Colaboratory
float bench_input[784] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,86.0,131.0,225.0,225.0,225.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,13.0,73.0,197.0,253.0,252.0,252.0,252.0,252.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.0,29.0,29.0,154.0,187.0,252.0,252.0,253.0,252.0,252.0,233.0,145.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,29.0,252.0,253.0,252.0,252.0,252.0,252.0,253.0,204.0,112.0,37.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,169.0,253.0,255.0,253.0,228.0,126.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,98.0,243.0,252.0,253.0,252.0,246.0,130.0,38.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,98.0,240.0,252.0,252.0,253.0,252.0,252.0,252.0,221.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,225.0,252.0,252.0,236.0,225.0,223.0,230.0,252.0,252.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,146.0,252.0,157.0,50.0,0.0,0.0,25.0,205.0,252.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,26.0,207.0,253.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,29.0,19.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,73.0,205.0,252.0,79.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,120.0,215.0,209.0,175.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,19.0,209.0,252.0,220.0,79.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,174.0,252.0,252.0,239.0,140.0,0.0,0.0,0.0,0.0,0.0,29.0,104.0,252.0,249.0,177.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,174.0,252.0,252.0,223.0,0.0,0.0,0.0,0.0,0.0,0.0,174.0,252.0,252.0,223.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,141.0,241.0,253.0,146.0,0.0,0.0,0.0,0.0,169.0,253.0,255.0,253.0,253.0,84.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,178.0,252.0,154.0,85.0,85.0,210.0,225.0,243.0,252.0,215.0,121.0,27.0,9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,66.0,208.0,220.0,252.0,253.0,252.0,252.0,214.0,195.0,31.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,19.0,37.0,84.0,146.0,223.0,114.0,28.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];

//--------------------------------------------------------------------
// Main Function
//--------------------------------------------------------------------
int main( int argc, char *argv[]){
  int i=0, ifeat, ofeat, y, x;

  float *in_img = (float *)malloc(28*28*sizeof(float));
  float *global_cb1_input = (float *)alignedMalloc(28*28*sizeof(float));
  int *out_img = (int *)alignedMalloc(1*sizeof(int));

  cl_int status;
  cl_event write_event, kernel_event;
  if (!init_opencl()) return -1;

  // load testbench
  for( i = 0; i < 784; i++){
    in_img[i] = bench_input[i];
  }

  // Call hardware
  status = clEnqueueWriteBuffer(queue, inp_buf, CL_FALSE, 0, 28*28*sizeof(float), (void *)global_cb1_input, 0, NULL, &write_event);
  checkError(status, "Failed to transfer input");
  status = clSetKernelArg(kernel,  0, sizeof(cl_mem), &inp_buf);
  status = clSetKernelArg(kernel,  1, sizeof(cl_mem), &out_buf);
  status = clEnqueueTask(queue, kernel, 1, &write_event, &kernel_event);
  status = clWaitForEvents(1, &kernel_event);
  checkError(status, "Failed to finish event");
  
  cl_ulong start, end;
  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
  status = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0,
            1*sizeof(int), out_img, 1, &kernel_event, NULL);
  checkError(status, "Failed to copy data from device");
  clReleaseEvent(write_event);
  clReleaseEvent(kernel_event);

  // output results
  printf("y[0] = %d\n", out_img[0]);

  // clean system
  free(in_img);
  alignedFree(global_cb1_input);
  alignedFree(out_img);
  cleanup();

  return 0;
}

//------ bool init_opencl -------{{{
bool init_opencl() 
{
  cl_int status;
  unsigned int num_devices;
  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  cl_device_id *ids = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  if (num_devices != 1) {
    printf("ERROR: invalid device number %d\n", num_devices);
    return false;
  }
  device = ids[0];

  // Create the context.
  context = clCreateContext(NULL, num_devices, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("dt_mnist", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Kernel.
  const char *kernel_name = "dt_mnist";
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

  // Input buffers.
  inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 28*28*sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for input");

  // Output buffer.
  out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1*sizeof(int), NULL, &status);
  checkError(status, "Failed to create buffer for output");

  return true;
}

void cleanup() {
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);
  clReleaseProgram(program);
  clReleaseContext(context);
}
