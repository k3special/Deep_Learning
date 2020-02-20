#!/usr/bin/env python

import cv2

from cifar10 import load_data


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape)

    ind = 0

    image = cv2.cvtColor(x_train[ind], cv2.COLOR_BGR2RGB)

    labels = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog',
              'Frog', 'Horse', 'Ship', 'Truck')

    print(labels[y_train[ind][0]])

    name = labels[y_train[ind][0]]

    cv2.imwrite('cifar_' + str(ind) + '_' + name + '.jpg', image)

    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
