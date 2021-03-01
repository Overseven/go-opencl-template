package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/MasterOfBinary/go-opencl/opencl"
)

const (
	deviceType = opencl.DeviceTypeAll

	dataSize = 128
)

// var (
// 	platform     opencl.Platform
// 	device       opencl.Device
// 	context      opencl.Context
// 	commandQueue opencl.CommandQueue
// 	program      opencl.Program
// 	kernelCode   string
// )

func printHeader(name string) {
	fmt.Println(strings.ToUpper(name))
	for _ = range name {
		fmt.Print("=")
	}
	fmt.Println()
}

func initCL() (opencl.Context, opencl.Device, error) {
	foundDevice := false
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		return opencl.Context{}, opencl.Device{}, err
	}

	var (
		name string
		//platform opencl.Platform
		device opencl.Device
	)

	for _, curPlatform := range platforms {
		err = curPlatform.GetInfo(opencl.PlatformName, &name)
		if err != nil {
			return opencl.Context{}, opencl.Device{}, err
		}

		devices, err := curPlatform.GetDevices(deviceType)
		if err != nil {
			return opencl.Context{}, opencl.Device{}, err
		}

		// Use the first available device
		if len(devices) > 0 && !foundDevice {
			var available bool
			err = devices[0].GetInfo(opencl.DeviceAvailable, &available)
			if err == nil && available {
				//platform := curPlatform
				device = devices[0]
				foundDevice = true
			}
		}
	}
	if !foundDevice {
		return opencl.Context{}, opencl.Device{}, errors.New("No device found")
	}

	context, err := device.CreateContext()
	if err != nil {
		return opencl.Context{}, opencl.Device{}, err
	}

	return context, device, nil
}

func initCommandQueue(context opencl.Context, device opencl.Device) (opencl.CommandQueue, error) {
	queue, err := context.CreateCommandQueue(device)
	if err != nil {
		return opencl.CommandQueue{}, err
	}
	return queue, nil
}

func initKernels(filename string, kernelNames []string, context opencl.Context, device opencl.Device) (map[string]opencl.Kernel, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	code, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}
	kernelCode := string(code)

	program, err := context.CreateProgramWithSource(kernelCode)
	if err != nil {
		return nil, err
	}
	var log string
	err = program.Build(device, &log)
	if err != nil {
		return nil, err
	}

	kernels := make(map[string]opencl.Kernel)
	for _, kname := range kernelNames {
		kernel, err := program.CreateKernel(kname)
		if err != nil {
			return nil, err
		}
		kernels[kname] = kernel
	}

	return kernels, nil
}

func main() {

	context, device, err := initCL()
	if err != nil {
		panic(err)
	}
	defer context.Release()

	commandQueue, err := initCommandQueue(context, device)
	if err != nil {
		panic(err)
	}
	defer commandQueue.Release()

	kernelFileName := "kernels/kernel.cl"
	kernelName := "kern"

	kernels, err := initKernels(kernelFileName, []string{kernelName}, context, device)
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
