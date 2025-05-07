from tensorflow.keras.callbacks import Callback

class SocketIOCallback(Callback):
    def __init__(self, socketio, session_id):
        super().__init__()
        self.socketio = socketio
        self.session_id = session_id

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.socketio.emit('epoch_update', {
                'epoch': epoch + 1,
                'loss': logs.get('loss', 0),
                'val_loss': logs.get('val_loss', 0)
            }, room=self.session_id)
