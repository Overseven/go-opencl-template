package gocl

import (
	"errors"
	"io/ioutil"
	"os"

	"github.com/MasterOfBinary/go-opencl/opencl"
)

func InitCL(devType opencl.DeviceType) (opencl.Context, opencl.Device, error) {
	foundDevice := false
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		return opencl.Context{}, opencl.Device{}, err
	}

	var (
		name   string
		device opencl.Device
	)

	for _, curPlatform := range platforms {
		err = curPlatform.GetInfo(opencl.PlatformName, &name)
		if err != nil {
			return opencl.Context{}, opencl.Device{}, err
		}

		devices, err := curPlatform.GetDevices(devType) // NOTE: choose right device type
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

func InitCommandQueue(context opencl.Context, device opencl.Device) (opencl.CommandQueue, error) {
	queue, err := context.CreateCommandQueue(device)
	if err != nil {
		return opencl.CommandQueue{}, err
	}
	return queue, nil
}

func InitKernels(filename string, kernelNames []string, context opencl.Context, device opencl.Device) (map[string]opencl.Kernel, error) {
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
