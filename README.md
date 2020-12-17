# IntelAdventCalender2020
A Decision Tree Accelerator on DE10-nano

- Intel OpenCL 18.1

## setup DE10-nano
Read a document from Terasic Inc.

https://www.terasic.com.tw/cgi-bin/page/archive_download.pl?Language=English&No=1046&FID=d646001078b0d07866d5aee4af0a0695

## Training Decision Tree on Google-Colaboratory
Access the following URL:

https://colab.research.google.com/drive/19bv_Xv807IIydt_e5srejC4BpPf60Dy_#scrollTo=N24VCi22d7SR

Download the following files:
- dt_mnist.cl
- bench_input.h

## Emulation (Optional)
$ aoc -march=emulator device/dt_mnist.cl -o bin/dt_mnist.aocx -board=de10nano

$ make

$ CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host

## Synthesis

$ aoc device/dt_mnist.cl -o bin/dt_mnist.aocx -board=de10nano

$ make

Copy bin/host and bin/dt_mnist.aocx to DE10-nano FPGA board, then execute it.
