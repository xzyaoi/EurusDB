class BaseModel(object):
    def __init__(self, name, page_size=1) -> None:
        super().__init__()
        self.name = name
        self.page_size = page_size

    def train(self, x_train, y_train, x_test, y_test):
        # x, y are numpy array
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
