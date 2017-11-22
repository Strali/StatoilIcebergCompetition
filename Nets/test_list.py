models = ['m1', 'm2', 'm3']

for i, clf in enumerate(models):
    if (clf == 'm1'):
        print('Using model {}'.format(str(clf)))
    elif (clf == 'm2'):
        print('Using model {}'.format(str(clf)))
    elif (clf == 'm3'):
        print('Using model {}'.format(str(clf)))
    else:
        print('Unspecified model')