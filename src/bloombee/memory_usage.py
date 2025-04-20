import torch
from pynvml.smi import nvidia_smi
from pynvml import *

def nvidia_smi_usage():
	logger = ''
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	# logger += "\n Nvidia-smi: " + str((info.used) / 1024 / 1024 / 1024) + " GB"
	return (info.used) / 1024 / 1024 / 1024

def see_memory_usage(message, force=True):
	logger = ''
	logger += message
	nvmlInit()

	# nvidia_smi.nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(handle)
	logger += "\n Nvidia-smi: " + str((info.used) / 1024 / 1024 / 1024) + " GB"
	# logger += message

	# logger += str(torch.cuda.memory_snapshot())
	# logger += str(torch.cuda.memory_summary(device=0, abbreviated=False))

	logger += '\n    Memory Allocated: '+str(torch.cuda.memory_allocated() / (1024 * 1024 * 1024)) +'  GigaBytes\n'
	logger +=   'Max Memory Allocated: ' + str(
		torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)) + '  GigaBytes\n'



	# logger +=       'Cache Allocated '+str(torch.cuda.memory_cached() / (1024 * 1024 * 1024))+'  GigaBytes\n'
	# logger +=       'Max cache Allocated '+str(torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))+'  GigaBytes\n'


	
	print(logger)


# input("Press Any Key To Continue ..")
