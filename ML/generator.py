import numpy as np
import argparse
import sys, os
import json
from mlutils import get_frames,center_frame,get_centroid_1D,get_centroid

input_shape = (51,51)

class ProgressBar(object):
    def __init__(self,n_steps,barlength=40):
        self.n_steps = n_steps
        self.barlength = barlength
        self._curr_progress = 0
        self._finished = False

    @property
    def _progress(self):
        return self._curr_progress/float(self.n_steps)

    def update(self):
        if self._finished:
            return
        self._curr_progress += 1
        status = ""
        if self._progress >= 1:
            self._curr_progress = self.n_steps
            self._finished=True
            status = "Done.\r\n"
        self.print_(status)

    def stop(self,message=""):
        self.print_("Stopped... {}\r\n".format(message))
    
    def print_(self,status=""):
        block = int(round(self.barlength*self._progress))
        text = "\rProgress: [{0}] {1:.0f}% {2}".format( "#"*block + "-"*(self.barlength-block), self._progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

def get_winner(frames):
    player=frames[:,:,:,0]
    players,counts = np.unique(player[-1],return_counts=True)
    order = counts.argsort()[::-1]
    target_id = players[order[0]]
    if target_id == 0 and len(order)>1:
        target_id = players[order[1]]
    return target_id

def get_chunks(total,max_replays):

    n_chunks = total/max_replays
    rest = total % max_replays

    if rest==0:
        return [total/n_chunks for _ in range(n_chunks)]
    
    n_chunks += 1
    n_per_chunk = int(total/float(n_chunks))+1

    chunks = []
    for _ in range(n_chunks):
        chunks.append(n_per_chunk if total >= n_per_chunk else total)
        total -= n_per_chunk

    return chunks

def main(replays_folder, output_folder, player, max_replays):

    if replays_folder[-1] == '/':
        replays_folder = replays_folder[:-1]

    out_dir = "{}/{}".format(output_folder,replays_folder.split('/')[-1])
    os.mkdir(out_dir)

    np.random.seed(0)

    replay_files = [fname for fname in os.listdir(replays_folder) if fname[-4:]=='.hlt']
    # replay_files = replay_files[:50]
    n_replays = len(replay_files)

    progbar = ProgressBar(n_replays)

    print('SELECTING')
    print('Loading')
    selected = []

    for replay_name in replay_files:

        progbar.update()

        replay = json.load(open('{}/{}'.format(replays_folder,replay_name)))

        frames = get_frames(replay)

        target_id = get_winner(frames)
        if target_id == 0: continue
        if player not in replay['player_names'][target_id-1].lower():
            continue

        selected.append(replay_name)

        del replay

    print("{}/{} replays have been selected.".format(len(selected),n_replays))

    print('Shuffling Replays')

    np.random.shuffle(selected)

    chunks = get_chunks(len(selected),max_replays)

    print('Processing replays')

    progbar2 = ProgressBar(len(selected))

    cursor = 0
    for i_chunk,chunk in enumerate(chunks):
        # print("Chunk {}".format(i_chunk))
        # print("{} to {}".format(cursor,cursor+chunk))

        training_input = []
        training_target = []

        test_input = []
        test_target = []

        for i,replay_name in enumerate(selected[cursor:cursor+chunk]):

            progbar2.update()

            is_training = i < 0.8*chunk

            replay = json.load(open('{}/{}'.format(replays_folder,replay_name)))
            frames = get_frames(replay)
            target_id = get_winner(frames)

            moves = np.array(replay['moves'])

            is_player = frames[:,:,:,0]==target_id
            filtered_moves = np.where(is_player[:-1],moves,np.zeros_like(moves))
            categorical_moves = (np.arange(5) == filtered_moves[:,:,:,None]).astype(int)
            
            wrapped_frames = np.empty(shape=(frames.shape[0],input_shape[0],input_shape[1],frames.shape[3]))
            wrapped_moves = np.empty(shape=(categorical_moves.shape[0],input_shape[0],input_shape[1],categorical_moves.shape[3]))

            iframes = np.empty(shape=frames.shape[:3]+(4,))

            iframes[:,:,:,0] = frames[:,:,:,0] == target_id
            iframes[:,:,:,1] = (frames[:,:,:,0] != target_id) & (frames[:,:,:,0] != 0)
            iframes[:,:,:,2] = frames[:,:,:,1]/20.
            iframes[:,:,:,3] = frames[:,:,:,2]/255.

            for frame,move in zip(iframes,categorical_moves):
                centroid = get_centroid(frame)
                wframe = center_frame(frame,centroid,wrap_size=input_shape)
                wmoves = center_frame(move,centroid,wrap_size=input_shape)
                if is_training:
                    training_input.append(wframe)
                    training_target.append(wmoves)
                else:
                    test_input.append(wframe)
                    test_target.append(wmoves)

            del replay

        np.save("{}/training_input_{}.npy".format(out_dir,i_chunk),training_input)
        np.save("{}/training_target_{}.npy".format(out_dir,i_chunk),training_target)
        np.save("{}/test_input_{}.npy".format(out_dir,i_chunk),test_input)
        np.save("{}/test_target_{}.npy".format(out_dir,i_chunk),test_target)

        cursor += chunk

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replays', type=str, help="Folder containing the replays")
    parser.add_argument('output', type=str, help="Output Folder")
    parser.add_argument('player', type=str, help="Player to copy")
    parser.add_argument('-m','--max_replays', type=int, default=200, help="Maximum number of replays per chunk")
    args = parser.parse_args()

    main(args.replays, args.output, args.player, args.max_replays)
