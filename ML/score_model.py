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

    mv_in_correct_2 = all_correct*(actual!=0)*(pred!=0)
    mv_acc_2 = np.nanmean((mv_in_correct_2.sum(axis=2).sum(axis=1).astype(float)/((actual!=0)*(pred!=0)).sum(axis=2).sum(axis=1).astype(float)))

    in_correct = all_correct*input[:,:,:,0]
    all_acc = (in_correct.sum(axis=2).sum(axis=1)/input[:,:,:,0].sum(axis=2).sum(axis=1)).mean()

    naive_acc = (all_correct.sum(axis=2).sum(axis=1)/(51*51.)).mean()

    # naive_acc = (((predictions>0.5)==target).sum(axis=3).sum(axis=2).sum(axis=1)/(51*51*5.)).mean()

    return still_acc, mv_acc, mv_acc_2, all_acc, naive_acc

def main(model_url, replays, replays_chunk, test_only, double_input):

    np.random.seed(0)

    print('Loading Features')

    if replays_chunk<0:

        all_files = [fname for fname in os.listdir(replays) if fname[-4:] == '.npy']

        # print(all_files)

        all_files.sort()

        # print(all_files)


        if not test_only:
            training_inputs = [np.load('{}/{}'.format(replays,fname)) for fname in all_files if 'training_input' in fname and 'alt' not in fname and 'prev' not in fname]
            training_targets = [np.load('{}/{}'.format(replays,fname)) for fname in all_files if 'training_target' in fname]

            training_input = np.concatenate(training_inputs)
            training_target = np.concatenate(training_targets)

            if double_input:
                training_inputs_alt = [np.load('{}/{}'.format(replays,fname)) for fname in all_files if 'training_input_alt' in fname]
                training_input_alt = np.concatenate(training_inputs_alt)

        test_inputs = [np.load('{}/{}'.format(replays,fname)) for fname in all_files if 'test_input' in fname and 'alt' not in fname and 'prev' not in fname]
        test_targets = [np.load('{}/{}'.format(replays,fname)) for fname in all_files if 'test_target' in fname]

        test_input = np.concatenate(test_inputs)
        test_target = np.concatenate(test_targets)

        if double_input:
            test_inputs_alt = [np.load('{}/{}'.format(replays,fname)) for fname in all_files if 'test_input_alt' in fname]
            test_input_alt = np.concatenate(test_inputs_alt)

    else:
        if not test_only:
            training_input = np.load('{}/training_input_{}.npy'.format(replays,replays_chunk))
            training_target = np.load('{}/training_target_{}.npy'.format(replays,replays_chunk))
            if double_input:
                training_input_alt = np.load('{}/training_input_alt_{}.npy'.format(replays,replays_chunk))

        test_input = np.load('{}/test_input_{}.npy'.format(replays,replays_chunk))
        test_target = np.load('{}/test_target_{}.npy'.format(replays,replays_chunk))
        if double_input:
            test_input_alt = np.load('{}/test_input_alt_{}.npy'.format(replays,replays_chunk))

    print('Done.')

    print('Loading Model')

    # with open(os.devnull, 'w') as sys.stderr:
    #     from keras.models import load_model
    #     model = load_model(model_url)

    from keras.models import load_model
    model = load_model(model_url)

    print('Done.')

    print('Making Predictions')

    if not test_only:
        if double_input:
            training_predictions = model.predict([training_input,training_input_alt])
        else:
            training_predictions = model.predict(training_input)

    if double_input:
        test_predictions = model.predict([test_input,test_input_alt])
    else:
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

    if not test_only:
        still_acc, mv_acc, mv_acc_2, all_acc, naive_acc = compute_accuracies(training_input, training_target, training_predictions)

        print('-'*40)
        print("TRAINING ACCURACY")
        print('-'*40)
        print("still: {:.2f}%".format(still_acc*100))
        print("move: {:.2f}%".format(mv_acc*100))
        print("move restricted: {:.2f}%".format(mv_acc_2*100))
        print("overall: {:.2f}%".format(all_acc*100))
        print("naive: {:.2f}%".format(naive_acc*100))
        print('-'*40)
        print('')

    still_acc, mv_acc, mv_acc_2, all_acc, naive_acc = compute_accuracies(test_input, test_target, test_predictions)

    print('-'*40)
    print("TEST ACCURACY")
    print('-'*40)
    print("still: {:.2f}%".format(still_acc*100))
    print("move: {:.2f}%".format(mv_acc*100))
    print("move restricted: {:.2f}%".format(mv_acc_2*100))
    print("overall: {:.2f}%".format(all_acc*100))
    print("naive: {:.2f}%".format(naive_acc*100))
    print('-'*40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('replays')
    parser.add_argument('-rc','--replays_chunk',type=int,default=-1)
    parser.add_argument('-t','--test',action='store_true')
    parser.add_argument('-di','--double_input', action="store_true", help="Use previous moves as additional input")

    args = parser.parse_args()

    main(
        args.model, 
        args.replays,
        args.replays_chunk,
        args.test,
        args.double_input
        )