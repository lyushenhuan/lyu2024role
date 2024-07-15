from sklearn.model_selection import KFold
from .layer import *
from .utils import *

class gcForest:
    def __init__(self, num_estimator, num_forests, num_classes, max_layer=100, max_depth=31, n_fold=5):
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.num_classes = num_classes
        self.layer_list = []
        self.number_of_layers = max_layer
        self.best_layer = -1

    def train(self, train_data, train_label):
        train_data_raw = train_data.copy()

        # return value
        val_p = []
        val_acc = []

        best_train_acc = 0.0
        layer_index = 0
        best_layer_index = 0
        bad = 0

        while layer_index < self.max_layer:
            # print("layer " + str(layer_index))
            layer = Layer(self.num_forests, self.num_estimator, self.num_classes,
                          self.n_fold, layer_index, self.max_depth, 1)
            val_prob, val_stack = layer.train(train_data, train_label)
            self.layer_list.append(layer)

            train_data = np.concatenate([train_data_raw, val_stack], axis=1)
            train_data = np.float16(train_data)
            train_data = np.float64(train_data)

            temp_val_acc = compute_accuracy(train_label, val_prob)
            val_acc.append(temp_val_acc)

            # if best_train_acc > temp_val_acc:
            #     bad += 1
            # else:
            #     bad = 0
            #     best_train_acc = temp_val_acc
            #     best_layer_index = layer_index
            # if bad >= 3:
            #     self.number_of_layers = layer_index+1
            #     break
            best_layer_index = layer_index

            layer_index = layer_index + 1
            self.best_layer = best_layer_index

        return [val_p, val_acc, best_layer_index]

    def train_and_predict(self, train_data, train_label, test_data, test_label):
        # basis information of dataset
        num_samples, num_features = train_data.shape

        # basis process
        train_data_raw = train_data.copy()
        test_data_raw = test_data.copy()

        # return value
        train_p = []
        train_acc = []
        val_p = []
        val_acc = []
        test_p = []
        test_acc = []

        best_train_acc = 0.0
        layer_index = 0
        best_layer_index = 0
        bad = 0

        while layer_index < self.max_layer:
            # print("layer " + str(layer_index))
            layer = Layer(self.num_forests, self.num_estimator, self.num_classes,
                          self.n_fold, layer_index, self.max_depth, 1)

            val_prob, val_stack, test_prob, test_stack = \
                layer.train_and_predict(train_data, train_label, test_data)

            self.layer_list.append(layer)

            train_data = np.concatenate([train_data_raw, val_stack], axis=1)
            test_data = np.concatenate([test_data_raw, test_stack], axis=1)
            train_data = np.float16(train_data)
            test_data = np.float16(test_data)
            train_data = np.float64(train_data)
            test_data = np.float64(test_data)

            temp_val_acc = compute_accuracy(train_label, val_prob)
            # print("val  acc:" + str(temp_val_acc))

            temp_test_acc = compute_accuracy(test_label, test_prob)
            # print("test acc:" + str(temp_test_acc))

            # val_p.append(val_prob)
            # test_p.append(test_prob)
            test_acc.append(temp_test_acc)
            val_acc.append(temp_val_acc)

            if best_train_acc > temp_val_acc:
                bad += 1
            else:
                bad = 0
                best_train_acc = temp_val_acc
                best_layer_index = layer_index
            if bad >= 3:
                self.number_of_layers = layer_index+1
                break

            layer_index = layer_index + 1
            self.best_layer = best_layer_index

        return [val_p, val_acc, test_p, test_acc, best_layer_index]

    def predict(self, test_data):
        test_data_raw = test_data.copy()
        layer_index = 0
        while layer_index <= self.best_layer:
            layer = self.layer_list[layer_index]
            predict_prob = np.zeros((self.num_forests, test_data.shape[0], self.num_classes), dtype=np.float64)
            n_dim = test_data.shape[1]
            for forest_index in range(self.num_forests):
                predict_prob_forest = np.zeros([test_data.shape[0], self.num_classes])
                for kfold in range(self.n_fold):
                    mf = layer.forest_list[forest_index][kfold]
                    predict_p = mf.predict_proba(test_data)
                    predict_prob_forest += predict_p
                predict_prob_forest /= self.n_fold
                predict_prob[forest_index, :] = predict_prob_forest
            predict_avg = np.sum(predict_prob, axis=0)
            predict_avg /= self.num_forests
            predict_concatenate = predict_prob.transpose((1, 0, 2))
            predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)

            test_prob, test_stack = predict_avg, predict_concatenate

            test_data = np.concatenate([test_data_raw, test_stack], axis=1)
            test_data = np.float16(test_data)
            test_data = np.float64(test_data)
            layer_index+=1

        return test_prob