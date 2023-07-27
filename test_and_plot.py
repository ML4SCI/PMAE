import torch
from models.masks import SpecificParticleMask, KinematicMask
import utils
from sklearn.metrics import roc_curve, auc, accuracy_score
import os
import numpy as np
import matplotlib.pyplot as plt

# Test loop
def test(loader, test_batch_size, X_test_arr, test_labels, names, models, device, mask, scaler, output_vars, information, model_name, lower=[0,-3.2,-1.6,0], upper=[4,3.2,1.6,3.2]):
    tae, classifier = models[0], models[1]
    tae.eval()
    classifier.eval()
    # Scale back to original scale
    X_test_arr = X_test_arr.reshape(X_test_arr.shape[0], X_test_arr.shape[1] * X_test_arr.shape[2] - 6 * (4 % output_vars))
    outputs_arrays = np.zeros((X_test_arr.shape[0], X_test_arr.shape[1], 6))
    X_test_arr_tensor = torch.tensor(X_test_arr)
    if information == 'partial':
        with torch.no_grad():
            all_preds = []
            for i in range(6):
                outputs_arr = torch.zeros(X_test_arr_tensor.size(0), X_test_arr_tensor.size(1))
                outputs_arr_2 = torch.zeros(X_test_arr_tensor.size(0))
                for batch_idx, batch in enumerate(loader):
                    # Move the data to the device
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if mask is not None:
                        if mask == 0:
                            mask_layer = SpecificParticleMask(4, i)
                        else:
                            mask_layer = KinematicMask(mask)
                        # Mask input data
                        masked_inputs = mask_layer(inputs)

                    # Forward pass
                    outputs = tae(masked_inputs)
                    outputs = torch.reshape(outputs, (outputs.size(0),
                                                      outputs.size(1) * outputs.size(2)))

                    masked_inputs = torch.reshape(masked_inputs, (masked_inputs.size(0),
                                                                  masked_inputs.size(1) * masked_inputs.size(2)))
                    outputs_2 = classifier(torch.cat((outputs, masked_inputs), axis=1)).squeeze(1)

                    outputs_arr[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size] = outputs
                    outputs_arr_2[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size] = outputs_2

                outputs_arr = outputs_arr.cpu().numpy()
                outputs_arr_2 = outputs_arr_2.cpu().numpy()

                fpr, tpr, _ = roc_curve(test_labels, outputs_arr_2)
                roc_auc = auc(fpr, tpr)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr, tpr, label='(ROC-AUC = {:.3f})'.format(roc_auc))
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                masked_parts = ['lepton', 'missing energy', 'jet 1', 'jet 2', 'jet 3', 'jet 4']
                plt.title('ROC curve masked ' + masked_parts[i])
                plt.legend(loc='best')
                plt.show()
                binary_preds = [1 if p > 0.5 else 0 for p in outputs_arr_2]
                acc = accuracy_score(test_labels, binary_preds)
                print('Classification Accuracy (masked ', masked_parts[i], '): ', acc)
                
                plt.savefig('./outputs/' + model_name + '/ROC_AUC_partial_' + masked_parts[i])

                X_test_arr_hh = X_test_arr[test_labels==1]
                X_test_arr_tt = X_test_arr[test_labels==0]
                outputs_arr_hh = outputs_arr[test_labels==1]
                outputs_arr_tt = outputs_arr[test_labels==0]

                # Generate scatter plots
                utils.make_hist2d(i, output_vars, 0, names, X_test_arr_hh, outputs_arr_hh, scaler, 'di-Higgs', './outputs/' + model_name, lower=lower[0], upper=upper[0])
                utils.make_hist2d(i, output_vars, 1, names, X_test_arr_hh, outputs_arr_hh, scaler, 'di-Higgs', './outputs/' + model_name, lower=lower[1], upper=upper[1])
                utils.make_hist2d(i, output_vars, 2, names, X_test_arr_hh, outputs_arr_hh, scaler, 'di-Higgs', './outputs/' + model_name, lower=lower[2], upper=upper[2])
                if output_vars == 4:
                    utils.make_hist2d(i, output_vars, 3, names, X_test_arr_hh, outputs_arr_hh, scaler, 'di-Higgs', './outputs/' + model_name, lower=lower[3], upper=upper[3])

                # Generate scatter plots
                utils.make_hist2d(i, output_vars, 0, names, X_test_arr_tt, outputs_arr_tt, scaler, 'ttbar', './outputs/' + model_name, lower=lower[0], upper=upper[0])
                utils.make_hist2d(i, output_vars, 1, names, X_test_arr_tt, outputs_arr_tt, scaler, 'ttbar', './outputs/' + model_name, lower=lower[1], upper=upper[1])
                utils.make_hist2d(i, output_vars, 2, names, X_test_arr_tt, outputs_arr_tt, scaler, 'ttbar', './outputs/' + model_name, lower=lower[2], upper=upper[2])
                if output_vars == 4:
                    utils.make_hist2d(i, output_vars, 3, names, X_test_arr_tt, outputs_arr_tt, scaler, 'ttbar', './outputs/' + model_name, lower=lower[3], upper=upper[3])

    elif information == 'full':
        with torch.no_grad():
            all_preds = []
            outputs_arr = torch.zeros(X_test_arr_tensor.size(0), X_test_arr_tensor.size(1))
            outputs_arr_2 = torch.zeros(X_test_arr_tensor.size(0))
            for batch_idx, batch in enumerate(loader):
                # Move the data to the device
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = torch.zeros(inputs.size(0), 6, output_vars).to(device)
                for i in range(6):
                    if mask is not None:
                        if mask == 0:
                            mask_layer = SpecificParticleMask(4, i)
                        else:
                            mask_layer = KinematicMask(mask)
                        # Mask input data
                        masked_inputs = mask_layer(inputs)
                    # Forward pass for autoencoder
                    temp_outputs = tae(masked_inputs)
                    outputs[:,i,:] = temp_outputs[:,i,:]                    

                outputs = torch.reshape(outputs, (outputs.size(0),
                                                  outputs.size(1) * outputs.size(2)))

                inputs = torch.reshape(inputs, (inputs.size(0),
                                                inputs.size(1) * inputs.size(2)))
                outputs_2 = classifier(torch.cat((outputs, inputs), axis=1)).squeeze(1)

                outputs_arr_2[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size] = outputs_2

            outputs_arr_2 = outputs_arr_2.cpu().numpy()

            fpr, tpr, _ = roc_curve(test_labels, outputs_arr_2)
            roc_auc = auc(fpr, tpr)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='(ROC-AUC = {:.3f})'.format(roc_auc))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve (full-information)')
            plt.legend(loc='best')
            plt.show()
            binary_preds = [1 if p > 0.5 else 0 for p in outputs_arr_2]
            acc = accuracy_score(test_labels, binary_preds)
            print('Classification Accuracy (full-information): ', acc)
            
            plt.savefig('./outputs/' + model_name + '/ROC_AUC_full')