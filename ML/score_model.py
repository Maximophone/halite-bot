import numpy as np
import argparse
import sys, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def compute_accuracies(input,target,predictions):
    pred = predictions.argmax(axis=3)
    actual = target.argmax(axis=3)

    still_all_correct = ((pred==0) == (actual==0))
    still_in_correct = still_all_correct*input[:,:,:,0]
    still_acc = (still_in_correct.sum(axis=2).sum(axis=1)/input[:,:,:,0].sum(axis=2).sum(axis=1)).mean()

    all_correct = (pred == actual)
    mv_in_correct = all_correct*(actual!=0)
    mv_acc = np.nanmean((mv_in_correct.sum(axis=2).sum(axis=1).astype(float)/(actual!=0).sum(axis=2).sum(axis=1).astype(float)))

    in_correct = all_correct*input[:,:,:,0]
    all_acc = (in_correct.sum(axis=2).sum(axis=1)/input[:,:,:,0].sum(axis=2).sum(axis=1)).mean()

    out_acc = (all_correct.sum(axis=2).sum(axis=1)/(51*51.)).mean()

    naive_acc = (((predictions>0.5)==target).sum(axis=3).sum(axis=2).sum(axis=1)/(51*51*5.)).mean()

    return still_acc, mv_acc, all_acc, out_acc, naive_acc

def main(model_url, replays, replays_chunk):

    np.random.seed(0)

    print('Loading Features')

    training_input = np.load('{}/training_input_{}.npy'.format(replays,replays_chunk))
    training_target = np.load('{}/training_target_{}.npy'.format(replays,replays_chunk))
    test_input = np.load('{}/test_input_{}.npy'.format(replays,replays_chunk))
    test_target = np.load('{}/test_target_{}.npy'.format(replays,replays_chunk))

    print('Done.')

    print('Loading Model')

    # with open(os.devnull, 'w') as sys.stderr:
    #     from keras.models import load_model
    #     model = load_model(model_url)

    from keras.models import load_model
    model = load_model(model_url)

    print('Done.')

    print('Making Predictions')

    training_predictions = model.predict(training_input)
    test_predictions = model.predict(test_input)

    print('Done.')

    # pred = test_predictions.argmax(axis=3)
    # actual = test_target.argmax(axis=3)

    # still_all_correct = ((pred==0) == (actual==0))
    # still_in_correct = still_all_correct*test_input[:,:,:,0]
    # still_acc = (still_in_correct.sum(axis=2).sum(axis=1)/test_input[:,:,:,0].sum(axis=2).sum(axis=1)).mean()

    # all_correct = (pred == actual)
    # mv_in_correct = all_correct*(actual!=0)
    # mv_acc = np.nanmean((mv_in_correct.sum(axis=2).sum(axis=1).astype(float)/(actual!=0).sum(axis=2).sum(axis=1).astype(float)))

    # in_correct = all_correct*test_input[:,:,:,0]
    # all_acc = (in_correct.sum(axis=2).sum(axis=1)/test_input[:,:,:,0].sum(axis=2).sum(axis=1)).mean()

    # out_acc = (all_correct.sum(axis=2).sum(axis=1)/(51*51.)).mean()

    # naive_acc = (((test_predictions>0.5)==test_target).sum(axis=3).sum(axis=2).sum(axis=1)/(51*51*5.)).mean()

    still_acc, mv_acc, all_acc, out_acc, naive_acc = compute_accuracies(training_input, training_target, training_predictions)

    print('-'*40)
    print("TRAINING ACCURACY")
    print('-'*40)
    print("still: {:.2f}%".format(still_acc*100))
    print("move: {:.2f}%".format(mv_acc*100))
    print("overall: {:.2f}%".format(all_acc*100))
    print("external: {:.2f}%".format(out_acc*100))
    print("naive: {:.2f}%".format(naive_acc*100))
    print('-'*40)
    print('')

    still_acc, mv_acc, all_acc, out_acc, naive_acc = compute_accuracies(test_input, test_target, test_predictions)

    print('-'*40)
    print("TEST ACCURACY")
    print('-'*40)
    print("still: {:.2f}%".format(still_acc*100))
    print("move: {:.2f}%".format(mv_acc*100))
    print("overall: {:.2f}%".format(all_acc*100))
    print("external: {:.2f}%".format(out_acc*100))
    print("naive: {:.2f}%".format(naive_acc*100))
    print('-'*40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('replays')
    parser.add_argument('-rc','--replays_chunk',type=int,default=0)

    args = parser.parse_args()

    main(
        args.model, 
        args.replays,
        args.replays_chunk
        )