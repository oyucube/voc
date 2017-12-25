class LOGGER:
    def __init__(self, log_file_path, file_id, log_mode=0):
        self.log_file_path = log_file_path
        self.out_file_name = file_id
        self.log_mode = log_mode
        self.log_buf = []
        self.val_array = np.full_like(np.zeros(n_epoch), np.nan)
        self.train_array = np.full_like(np.zeros(n_epoch), np.nan)
        self.loss_array = np.full_like(np.zeros(n_epoch), np.nan)
        self.max_acc = 0
        self.best = ""

        with open(log_file_path + "log.txt", "w") as f:
            f.write(" ")

    def l_print(self, str):
        print(str)
        self.log_buf.append(str)
        return

    def update_log(self):
        with open(log_file_path + "log.txt", "a") as f:
            for str in self.log_buf:
                f.write(str)
        return

    def set_loss(self, loss, epoch):
        self.loss_array[epoch] = loss

    def set_acc(self, train, val, epoch):
        self.train_array[epoch] = train
        self.train_array[epoch] = val
        self.best = ""
        if val > max_acc:
            self.max_acc = val
            self.best = "best"

    def save_acc(self):
        np.save(self.log_file_path + file_id + "test.npy", self.val_array)
        np.save(self.log_file_path + file_id + "train.npy", self.train_array)
        plt.figure()
        plt.ylim([0, 1])
        p1 = plt.plot(self.val_array, color="green")
        p2 = plt.plot(self.train_array, color="blue")
        plt.legend((p1[0], p2[0]), ("test", "train"), loc=2)
        plt.savefig(self.log_file_path + file_id + "acc.png")
        plt.figure()
        plt.plot(loss_array)
        plt.savefig(self.log_file_path + file_id + "loss.png")
        plt.close("all")