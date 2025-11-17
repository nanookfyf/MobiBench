from sentence_transformers import SentenceTransformer, util

# 轻量级模型选择
models = {
    'mini': "/Users/fengyunfei/Desktop/mobiagent/MobiBench/utils/models/weights/paraphrase-MiniLM-L6-v2",  # 80MB
}

model = SentenceTransformer(models['mini'])

def semantic_similarity(text1, text2):
    # 编码句子
    embeddings = model.encode([text1, text2])
    
    # 计算余弦相似度
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    # 或者使用点积相似度
    dot_score = util.dot_score(embeddings[0], embeddings[1]).item()
    
    return {
        'cosine_similarity': cosine_score,
        'dot_score': dot_score
    }

# 批量处理
def batch_similarity(texts1, texts2):
    embeddings1 = model.encode(texts1)
    embeddings2 = model.encode(texts2)
    return util.cos_sim(embeddings1, embeddings2)


if __name__ == "__main__":
    text_a = "三米粥屋"
    text_b = "三米粥铺"
    
    scores = semantic_similarity(text_a, text_b)
    print(f"Cosine Similarity: {scores['cosine_similarity']}")
    print(f"Dot Score: {scores['dot_score']}")
