import matplotlib.pyplot as plt
class ErrorTracker:
    def __init__(self) -> None:
        self.errors = {}
        self.count = 0
    
    def add_error(self, error: str):
        l = self.errors.get(error)
        if l is None:
            self.errors[error] = 1
        else:
            self.errors[error] = l + 1

        self.count+=1

    def plot_errors(self):
        labels = self.errors.keys()
        values = self.errors.values()
        plt.hist(values, label=labels)