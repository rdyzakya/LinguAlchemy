EPOCHES=30
OUT_DIR="./out"  # set customised out_path
EVAL_DIR="./eval"
SCALE=10 # Based on paper
# Define array of vectors
VECTORS=(
    'uriel/id' 'uriel/geo' 'uriel/inventory_knn' 'uriel/inventory_average'
    'uriel/syntax_knn' 'uriel/phonology_knn' 'uriel/fam'
    'ppl/ppl_vectors_minmax_scale' 'ppl/ppl_vectors_std_scale' 'uriel_plus/genetic'
    'uriel_plus/inventory' 'uriel_plus/morphological' 'uriel_plus/featural'
    'uriel_plus/phonological' 'uriel_plus/geographic' 'uriel_plus/syntactic'
    'uriel_plus_ppl/genetic_minmax_scaled' 'uriel_plus_ppl/geographic_std_scaled'
    'uriel_plus_ppl/morphological_minmax_scaled' 'uriel_plus_ppl/inventory_minmax_scaled'
    'uriel_plus_ppl/geographic_minmax_scaled' 'uriel_plus_ppl/morphological_std_scaled'
    'uriel_plus_ppl/inventory_std_scaled' 'uriel_plus_ppl/featural_minmax_scaled'
    'uriel_plus_ppl/phonological_minmax_scaled' 'uriel_plus_ppl/phonological_std_scaled'
    'uriel_plus_ppl/syntactic_std_scaled' 'uriel_plus_ppl/genetic_std_scaled'
    'uriel_plus_ppl/featural_std_scaled' 'uriel_plus_ppl/syntactic_minmax_scaled'
    'uriel_ppl/syntax_knn_std_scaled' 'uriel_ppl/inventory_knn_minmax_scaled'
    'uriel_ppl/id_minmax_scaled' 'uriel_ppl/phonology_knn_minmax_scaled'
    'uriel_ppl/inventory_average_std_scaled' 'uriel_ppl/inventory_knn_std_scaled'
    'uriel_ppl/syntax_knn_minmax_scaled' 'uriel_ppl/geo_std_scaled'
    'uriel_ppl/phonology_knn_std_scaled' 'uriel_ppl/fam_std_scaled'
    'uriel_ppl/inventory_average_minmax_scaled' 'uriel_ppl/id_std_scaled'
    'uriel_ppl/geo_minmax_scaled' 'uriel_ppl/fam_minmax_scaled'
)
# 
for MODEL_NAME in bert-base-multilingual-cased xlm-roberta-base;
do
    for VECTOR in "${VECTORS[@]}";
    do
        CUDA_VISIBLE_DEVICES=4,5 python -m src.lingualchemy \
        --model_name ${MODEL_NAME} --epochs ${EPOCHES}  \
        --out_path ${OUT_DIR}/massive/${MODEL_NAME}/scale${SCALE}_${VECTOR} \
        --vector ${VECTOR} --scale ${SCALE} --eval_path ${EVAL_DIR}/massive/${MODEL_NAME}_scale${SCALE}
    done
done
