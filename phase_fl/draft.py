class CustomCallback(keras.callbacks.Callback):
		def __init__(self):
				super(CustomCallback, self).__init__()
				self.history = {}
		
		def on_epoch_end(self, epoch, logs=None):
				if logs:
						for key in logs.keys():
								if epoch == 0:
										self.history[key] = []
								self.history[key].append(logs[key])
						print(self.history)
						
class HistoryPrintCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(HistoryPrintCallback, self).__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
    		print(logs.keys())

model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=2,
    verbose=0,
    validation_split=0.5,
    callbacks=[CustomCallback()],
)