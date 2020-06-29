import os, glob
import torch


def load_checkpoint(encoder, decoder, optimizer, device, args, is_train=True):

    pkl_list = glob.glob(os.path.join(args.model_dir, args.model, "*.pkl"))
    if len(pkl_list) == 0:
        print("No checkpoints available...")
        return encoder, decoder, optimizer, 0

    checkpoint = torch.load(pkl_list[-1], map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    if is_train:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return encoder, decoder, optimizer, checkpoint["epoch"]
