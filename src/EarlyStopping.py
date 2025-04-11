import io
import zipfile
import torch
import copy

class EarlyStopping:
    def __init__(self,
                 patience=10,
                 verbose=False,
                 delta=0,
                 zip_file="checkpoint.zip",
                 path='checkpoint.pth'):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.zip_file = zip_file
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None
        self.buffer = io.BytesIO()

    def __call__(self, val_loss, epoch, model, optimizer, scheduler):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler)
        elif val_loss < self.best_loss - self.delta:
            self.save_checkpoint(val_loss, epoch, model, optimizer, scheduler)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, epoch, model, optimizer, scheduler):
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...")

        #reduced_opt_state = self.reduce_optimizer_state(optimizer)

        checkpoint = {
            'model_state': model.state_dict()
        #    'optimizer_state': reduced_opt_state,
        #    'scheduler_state': scheduler.state_dict(),  # Save scheduler state
        #    'val_loss': val_loss,
        #    'epoch': epoch
        }

        #self.buffer = io.BytesIO()  # Reset buffer
        #torch.save(checkpoint, self.buffer)
        #self.buffer.seek(0)

        #with zipfile.ZipFile(self.zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        #    zipf.writestr(self.path, self.buffer.read())

        self.best_model_wts = checkpoint["model_state"]

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)

    def reduce_optimizer_state(self, optimizer):
        opt_state = copy.deepcopy(optimizer.state_dict())
        for state in opt_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.half()
        return opt_state

    def load_checkpoint(self, model, optimizer, scheduler, device='cuda'):
        with zipfile.ZipFile(self.zip_file, 'r') as zipf:
            with zipf.open(self.path) as f:
                checkpoint = torch.load(f, map_location=device)

        model.load_state_dict(checkpoint['model_state'])

        opt_state = checkpoint['optimizer_state']
        for state in opt_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.float()
        optimizer.load_state_dict(opt_state)

        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state'])  # Load scheduler state

        val_loss = checkpoint['val_loss']
        epoch = checkpoint['epoch']

        self.best_loss = val_loss

        print(f"Checkpoint loaded. Validation Loss: {val_loss:.6f}, Epoch: {epoch}")

        return model, optimizer, scheduler, epoch
