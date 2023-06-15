import re
import matplotlib.pyplot as plt

evaluation_metrics = ["precision", "recall", "F1"]


def remove_lines_with_keyword(input_file, output_file, keyword):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if keyword not in line:
                file.write(line)


# Exemple d'utilisation : suppression des lignes contenant le mot "annotators"
# remove_lines_with_keyword(
#     '15res_BiMRC(6).log', '15_res_new.log', 'annotators')

# Fonction pour extraire les époques et les précisions des données de test à partir d'un fichier de log
def extract_test_precisions(file_path, metric):
    epochs = []
    test_precisions = []

    with open(file_path, 'r') as file:
        log_data = file.readlines()
    epoch = 0
    for i in range(len(log_data)):
        if 'test' in log_data[i]:
            precision_line = log_data[i+1]
            if metric == "precision":
                precision_match = re.search(
                    r'Triplet - Precision: ([0-9.]+)', precision_line)
            elif metric == "recall":
                precision_match = re.search(
                    r'Recall: ([0-9.]+)', precision_line)
            elif metric == "F1":
                precision_match = re.search(
                    r'F1: ([0-9.]+)', precision_line)
            if precision_match:
                #epoch_match = re.search(r'Epoch:\[(\d+)', log_data[i])
                #if epoch_match:
                epoch += 1
                precision = float(precision_match.group(1))
                epochs.append(epoch)
                test_precisions.append(precision)

    return epochs, test_precisions

remove_lines_with_keyword(
     '14res_BiMRC.log', "15_res_BiMRC_original.log", 'annotators')
remove_lines_with_keyword('ours_14res_BiMRC.log', '15_res_ours.log', 'annotators')

# Lire les données à partir des fichiers de log
epochs1, test_precisions1 = extract_test_precisions(
     '15_res_BiMRC_original.log', evaluation_metrics[2])
epochs2, test_precisions2 = extract_test_precisions(
    '15_res_ours.log', evaluation_metrics[2])

# Tracer les courbes de précision pour les deux jeux de données
plt.plot(epochs1, test_precisions1, marker=None,
          label='RoBMRC')
plt.plot(epochs2, test_precisions2, marker=None,
         label='Ours')
plt.xlabel('Époque')
plt.ylabel(evaluation_metrics[2])
plt.title('Precision on 14res dataset')
plt.grid(True)
plt.legend()
plt.show()
