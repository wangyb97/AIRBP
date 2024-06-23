from AIRBP.Features import generateBPFeatures
from AIRBP.featureSelection import cife
from AIRBP.metricsPlot import shap_interaction_scatter
bp_features = generateBPFeatures(sequences, PGKM=True)

# Filter the original features
refined_features = cife(bp_features, label, num_features=10)

# Performance visualization of SVM using AIRBP
clf = SVC(probability=True)
shap_interaction_scatter(refined_features, label, clf=clf, sample_size=(0, 100), feature_size=(0, 10), image_path='./')