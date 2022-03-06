import run
import torch
import sys


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        sys.stdout.write("You are using a CUDA device.\n\n")
        torch.backends.cudnn.benchmark = True
        cur_device = torch.cuda.current_device()
        sys.stdout.write("Device:      \n")
        sys.stdout.write("    Index:      %s\n" % cur_device)
        sys.stdout.write("    Name:       %s\n" % torch.cuda.get_device_name(cur_device))
        sys.stdout.write("Versions:       \n")
        sys.stdout.write("    CUDA:       %s\n" % torch.version.cuda)
        sys.stdout.write("    cuDNN:      %s\n" % torch.backends.cudnn.version())
        sys.stdout.write("Memory Usage:\n")
        sys.stdout.write("    Max Alloc:  %g GB\n" % round(torch.cuda.max_memory_allocated(cur_device) / 1024 ** 3, 1))
        sys.stdout.write("    Allocated:  %g GB\n" % round(torch.cuda.memory_allocated(cur_device) / 1024 ** 3, 1))
        sys.stdout.write("    Cached:     %g GB\n" % round(torch.cuda.memory_reserved(cur_device) / 1024 ** 3, 1))
        sys.stdout.write("\n")
    else:
        device = 'cpu'
        sys.stdout.write("You are using CPU.\n")

    method = "2"
    data_path = "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    sys.stdout.write("You chose the method %s.\n" % method)
    sys.stdout.write("Data path: %s\n" % data_path)
    sys.stdout.write('\n')

    run.init(device, data_path, "method_" + method)
    run.train()
    accuracy = run.evaluate("valid")
    sys.stdout.write("Accuracy: %.2f%%\n" % (accuracy * 100))
