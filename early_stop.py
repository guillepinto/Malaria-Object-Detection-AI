import os

class EarlyStopping:
    def __init__(self, mode='max', patience=5, verbose=False, delta=0, accelerator=None, output_dir=None):
        """
        Args:
            mode (str): 'max' for metrics that need to be maximized, 'min' for metrics that need to be minimized.
            patience (int): Number of epochs to wait after the last improvement.
            verbose (bool): If True, prints messages when there are improvements or increments in the counter.
            delta (float): Minimum changes in the metric to consider an improvement.
            accelerator (Accelerator or None): Instance of Accelerator to save the state.
            output_dir (str or None): Directory to save the best model state.
        """
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accelerator = accelerator
        self.output_dir = output_dir

        if self.mode == 'max':
            self.is_better = lambda current, best: current > best + self.delta
        elif self.mode == 'min':
            self.is_better = lambda current, best: current < best - self.delta
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, score, epoch):
        if self.early_stop:
            return  # Do nothing if early_stop has already been triggered

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, epoch)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(score, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, epoch):
        '''Saves the model state when the metric improves'''
        if self.verbose:
            print(f'Best metric found: {score:.4f}. Saving model...')
        if self.output_dir is not None:
            output_dir_epoch = f"epoch_{epoch}"
            output_dir = os.path.join(self.output_dir, output_dir_epoch)
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            if self.accelerator is not None:
                self.accelerator.save_state(output_dir)
            else:
                # Optional: save the model state manually if there is no accelerator
                pass
                # Example: torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
