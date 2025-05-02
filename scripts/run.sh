EPOCHES=30
OUT_DIR="./out"  # set customised out_path
EVAL_DIR="./eval"
SCALE=10 # Based on paper
# Define array of vectors
VECTORS=(
    'uriel_ppl/geo_minmax_scaled' 
    'uriel_ppl/geo_std_scaled'
    'uriel_ppl/syntax_knn_minmax_scaled' 
    'uriel_ppl/syntax_knn_std_scaled' 
    'uriel_plus_ppl/syntactic_minmax_scaled'
    'uriel_plus_ppl/syntactic_std_scaled' 
    'uriel_plus_ppl/geographic_minmax_scaled' 
    # 'uriel_plus_ppl/geographic_std_scaled'
    'uriel_plus/syntactic'
    # 'uriel_plus/geographic' 
    # 'ppl/ppl_vectors_std_scale' 
    # 'ppl/ppl_vectors_minmax_scale' 
    # 'uriel/syntax_knn' 
    # 'uriel/geo' 
)

# 
for MODEL_NAME in xlm-roberta-base bert-base-multilingual-cased;
do
    for VECTOR in "${VECTORS[@]}";
    do
        CUDA_VISIBLE_DEVICES=2 python -m src.lingualchemy \
        --model_name ${MODEL_NAME} --epochs ${EPOCHES}  \
        --out_path ${OUT_DIR}/massive/${MODEL_NAME}/scale${SCALE}_${VECTOR} \
        --vector ${VECTOR} --scale ${SCALE} --eval_path ${EVAL_DIR}/massive/${MODEL_NAME}_scale${SCALE}
    done
done
