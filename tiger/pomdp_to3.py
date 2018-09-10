from model.pomdp_from import POMDP


class Tiger(POMDP):
    def __init__(self):
        super().__init__(3, 4, 2)
        self.t[0, 0, 0] = 1
        self.t[0, 1, :] = [0.5, 0.5, 0]
        self.t[1, 0, :] = [0.5, 0.5, 0]
        self.t[1, 1, 1] = 1
        self.t[:-2, -1, -1] = 1
        self.t[-2:, :, -1] = 1

        self.r[2, :2] = [10, -100]
        self.r[3, :2] = [-100, 10]
        self.r[:-2, :2] = -1

        # self.o[:2, 0, :] = [0.8, 0.2]
        # self.o[:2, 1, :] = [0.2, 0.8]
        self.o[0, 0, :] = [0.8, 0.2]
        self.o[0, 1, :] = [0.5, 0.5]
        self.o[1, 0, :] = [0.5, 0.5]
        self.o[1, 1, :] = [0.2, 0.8]
        # self.o[0, 0, :] =
        self.o[:2, 2, :] = 0.5
        self.o[2:, :, :] = 0.5
        # print(self.o)
        # exit()

        self.pre_calc()
