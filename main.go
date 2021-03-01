package main

import (
	"fmt"
	"strings"

	"github.com/MasterOfBinary/go-opencl/opencl"
	"github.com/Overseven/go-opencl-template/gocl"
)

const (
	deviceType = opencl.DeviceTypeAll

	dataSize = 128
)

func printHeader(name string) {
	fmt.Println(strings.ToUpper(name))
	for _ = range name {
		fmt.Print("=")
	}
	fmt.Println()
}

func main() {

	context, device, err := gocl.InitCL(opencl.DeviceTypeGPU)
	if err != nil {
		panic(err)
	}
	defer context.Release()

	commandQueue, err := gocl.InitCommandQueue(context, device)
	if err != nil {
		panic(err)
	}
	defer commandQueue.Release()

	kernelFileName := "kernels/kernel.cl"
	kernelName := "kern"

	kernels, err := gocl.InitKernels(kernelFileName, []string{kernelName}, context, device)
	if err != nil {
		panic(err)
	}

	for _, k := range kernels {
		defer k.Release()
	}

	buffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dataSize*4)
	if err != nil {
		panic(err)
	}
	defer buffer.Release()

	kern, _ := kernels[kernelName]

	err = kern.SetArg(0, buffer.Size(), &buffer)
	if err != nil {
		panic(err)
	}

	err = commandQueue.EnqueueNDRangeKernel(kern, 1, []uint64{dataSize})
	if err != nil {
		panic(err)
	}

	commandQueue.Flush()
	commandQueue.Finish()

	data := make([]float32, dataSize)

	err = commandQueue.EnqueueReadBuffer(buffer, true, data)
	if err != nil {
		panic(err)
	}

	fmt.Println()
	printHeader("Output")
	for _, item := range data {
		fmt.Printf("%v ", item)
	}
	fmt.Println()
}
