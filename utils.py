import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
import sys



class DataLoader():
    def __init__(self, data_dir = [] , row_transform = False, delimiter = ','):
        self.data_dir = data_dir
        self.rt = row_transform
        self.dt = delimiter
        self.num_dataset = len(data_dir)
        self.data = []
        self.data_val = []
        self.max_num_peds = 0
        self.max_peds_in_frame = np.zeros(len(data_dir), dtype=np.int32)
        self.a, self.b, self.c, self.d = 0,0,0,0
        pass

    def preprocess(self):

        for idx, directory in enumerate(self.data_dir):
            raw_data = np.genfromtxt(directory, delimiter=self.dt, dtype=np.float32)
            print(raw_data.shape)
            # return
            if self.rt:
                raw_data = np.transpose(raw_data)
            self.a = raw_data[2].min()
            self.b = raw_data[2].max()
            self.c = raw_data[3].min()
            self.d = raw_data[3].max()
            num_peds = int(np.max(raw_data, axis=1)[1])
            if self.max_num_peds < num_peds:
                self.max_num_peds = num_peds
            frames = np.unique(raw_data[0,:]).astype(np.int32)
            ped_ids = np.zeros([len(frames), num_peds], dtype=np.int32)
            data = np.zeros([num_peds, len(frames), 2])
            data_val = np.zeros([num_peds, len(frames)])
            for frame_idx,frame in enumerate(frames):
                num_peds_in_frame = len(raw_data[1,raw_data[0,:]==frame])
                if(num_peds_in_frame>self.max_peds_in_frame[idx]):
                    self.max_peds_in_frame[idx] = num_peds_in_frame
                ped_ids[frame_idx, :num_peds_in_frame] = raw_data[1, raw_data[0, :] == frame]
                data[ped_ids[frame_idx, :num_peds_in_frame]-1, frame_idx, :] = raw_data[2:, raw_data[0,:]==frame].T
                data_val[ped_ids[frame_idx, :num_peds_in_frame]-1, frame_idx] = 1

            for ped in range(num_peds):
                first_non_zero = 0
                first_idx = 0
                for frame in range(len(frames)):
                    if data_val[ped, frame] == 1 and first_non_zero == 0:
                        first_non_zero = 1
                        first_idx = frame
                    if data_val[ped, frame] == 1 and first_non_zero == 1:
                        data[ped, (first_idx + 1):frame, 0] = np.linspace(data[ped, first_idx, 0], data[ped, frame, 0],
                                                                        frame - first_idx + 1)[1:-1]
                        data[ped, (first_idx + 1):frame, 1] = np.linspace(data[ped, first_idx, 1], data[ped, frame, 1],
                                                                        frame - first_idx + 1)[1:-1]
                        data_val[ped, (first_idx + 1):frame] = 1
                        first_idx = frame
            f = open('data_'+str(idx), 'wb')
            pickle.dump([data, data_val],f, protocol=2)
            f.close()

    def load_preprocess(self, dataset_idx):

        for i in dataset_idx:
            f = open('data_'+str(i), 'rb')
            [data, data_val] = pickle.load(f)
            self.data.append(data)
            self.data_val.append(data_val)
            f.close()

    def visualize(self, dataset_idx_list):
        fig = plt.figure()
        plt.axis('equal')
        plt.grid()
        ax = fig.add_subplot(111)
        ax.set_xlim(self.a, self.b)
        ax.set_ylim(self.c, self.d)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black")
        peds_line = []
        peds_dot = []
        peds_line2 = []
        peds_dot2 = []
        max_peds = self.data[0].shape[0]
        color = np.random.rand(3, max_peds)
        for dataset_idx in dataset_idx_list:
            anim_running = True
            data = self.data[dataset_idx]
            data_val = self.data_val[dataset_idx]
            # max_peds = data.shape[0]
            max_frames = data.shape[1]

            
            # print(data.shape)
            # a = np.amax(data[:,2],axis=2)
            # b = np.amin(data,axis=2)
            # c = np.amax(data,axis=3)
            # d = np.amin(data,axis=3)
            
            
            
            for i in range(max_peds):
                if dataset_idx==0:
                    temp = ax.plot([], [],'--', lw=1,label = str(i), c = color[:,i])
                    peds_line.extend(temp)
                    temp = ax.plot([], [],'p', lw=1, label=str(i), c=color[:,i])
                    peds_dot.extend(temp)
                else :
                    temp = ax.plot([], [],'+', lw=1,label = str(i), c = color[:,i])
                    peds_line2.extend(temp)
                    temp = ax.plot([], [],'o', lw=1, label=str(i), c=color[:,i])
                    peds_dot2.extend(temp)


        fig.subplots_adjust(top=0.8)

        def init():
            for ped_line in peds_line:
                ped_line.set_data([], [])
            for ped_dot in peds_dot:
                ped_dot.set_data([], [])
            return peds_line,peds_dot,

        def animate(i):
            print('frame:', i, 'from: ', max_frames)
            for ped_num, ped_line in enumerate(peds_line):
                if self.data_val[0][ped_num,i] == 0:
                    ped_line.set_data([], [])
                    peds_dot[ped_num].set_data([],[])
                else:
                    (x,y) = ped_line.get_data()
                    ped_line.set_data(np.hstack((x,self.data[0][ped_num,i,0])), np.hstack((y[:],self.data[0][ped_num,i,1])))
                    peds_dot[ped_num].set_data(self.data[0][ped_num,i,0], self.data[0][ped_num,i,1])

            for ped_num, ped_line in enumerate(peds_line2):
                if self.data_val[1][ped_num,i] == 0:
                    ped_line.set_data([], [])
                    peds_dot2[ped_num].set_data([],[])
                else:
                    (x,y) = ped_line.get_data()
                    ped_line.set_data(np.hstack((x,self.data[1][ped_num,i,0])), np.hstack((y[:],self.data[1][ped_num,i,1])))
                    peds_dot2[ped_num].set_data(self.data[1][ped_num,i,0], self.data[1][ped_num,i,1])
                    
            return peds_line, peds_dot, peds_line2, peds_dot2

        # You can pause the animation by clicking on it.
        def onClick(event):
            nonlocal anim_running
            if anim_running:
                anim.event_source.stop()
                anim_running = False
            else:
                anim.event_source.start()
                anim_running = True
        fig.canvas.mpl_connect('button_press_event', onClick)
        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init,
                                       frames=max_frames,
                                       interval=100)
        plt.show()

def main(arg):
    print(arg)
    # return
    if arg==1:
        vis = DataLoader(data_dir=['./data/eth/hotel/pixel_pos.csv',
                                './data/eth/univ/pixel_pos.csv',
                                './data/ucy/univ/pixel_pos.csv',
                                './data/ucy/zara/zara01/pixel_pos.csv',
                                './data/ucy/zara/zara02/pixel_pos.csv'])
    else:
        # vis = DataLoader(data_dir=['./data/stgcnn/preds (1).csv','./data/stgcnn/gt (1).csv'], row_transform=True, delimiter=',')
        vis = DataLoader(data_dir=['./data/orca/circle/file.txt'], row_transform=True, delimiter=',')
    # Uncomment vis.preprocess() in the first run.
    vis.preprocess()
    vis.load_preprocess([0])
    # vis.load_preprocess([0])
    vis.visualize([0])

if __name__=="__main__":
    main(sys.argv[1])
    # vis = DataLoader(data_dir=['./data/eth/hotel/pixel_pos.csv',
    #                             './data/eth/univ/pixel_pos.csv',
    #                             './data/ucy/univ/pixel_pos.csv',
    #                             './data/ucy/zara/zara01/pixel_pos.csv',
    #                             './data/ucy/zara/zara02/pixel_pos.csv'])
    # # Uncomment vis.preprocess() in the first run.
    # vis.preprocess()
    # vis.load_preprocess([0, 1, 2, 3, 4])
    # vis.visualize(0)
    