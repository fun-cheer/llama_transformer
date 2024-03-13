#include <stdio.h>
#include <string.h>
#include <math.h>

const float EPS = 1e-5f;

typedef struct {
    int batch;
    int seq_len;
    int hidden_size;
    int intern_size;
    int heads;
    int kv_heads;
    int kv_hidden_size;
    int vocab_size;
    int layers;
} llama_config;

typedef struct {
    float *rms1;    // [layers, hidden_size]
    float *q;       // [layers, hidden_size, hidden_size]
    float *k;       // [layers, hidden_size, kv_hidden_size]
    float *v;       // [layers, hidden_size, kv_hidden_size]
    float *proj;    // [layers, hidden_size, hidden_size]
    float *rms2;    // [layers, hidden_size]
    float *ffn1;    // [layers, hidden_size, intern_size]
    float *ffn2;    // [layers, hidden_size, intern_size]
    float *ffn3;    // [layers, intern_size, hidden_size]
    float *rms3;    // [1, hidden_size]
    float *logits;  // [hidden_size, vocab_size]

    float *vocab;   // [vocab_size, hidden_size]
} llama_weight;

typedef struct {
    float *token;   // [1, hidden_size]
    float *rms1;    // [1, hidden_size]
    float *q;       // [1, hidden_size]
    float *kcache;  // [layers, seq_len, kv_hidden_size]
    float *vcache;  // [layers, seq_len, kv_hidden_size]
    float *atto;    // [1, hidden_size]
    float *proj;    // [1, hidden_size]
    float *rms2;    // [1, hidden_size]
    float *ffn1;    // [1, intern_size]
    float *ffn2;    // [1, intern_size]
    float *silu;    // [1, intern_size]
    float *ffn3;    // [1, hidden_size]
    float *rms3;    // [1, hidden_size]
    float *logits;  // [1, vocab_size]
} llama_io;

void rmsnorm(float *out, float *in, float *weight, int hidden_size) {
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        sum += in[i] * in[i];
    }
    sum /= hidden_size;
    sum += EPS;
    sum = 1.0f / sqrtf(sum);

    for (int i = 0; i < hidden_size; i++) {
        out[i] = weight[i] * (sum * in[i]);
    }
}

void matmul(float* out, float* A, float* B, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                out[i * n + j] = A[i * k + kk] * B[kk * n + j];
            }
        }
    }
}

void softmax(float *in, int size) {
    float max_val = in[0];
    for (int i = 1; i < size; i++) {
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        in[i] = expf(in[i] - max_val);
        sum += in[i];
    }

    sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        in[i] *= sum;
    }
}

// seq_len <= 2048
// head <= 32
void attention(float *out, float *q, float *kcache, float *vcache, llama_config *config) {
    float *buf = (float *)malloc(2048 * 32 * sizeof(float));
    int dim = config->hidden_size / config->heads;
    int roll = config->heads / config->kv_heads;

    for (int i = 0; i < config->heads; i++) {
        float *qi = q + i * dim;
        float *bufi = buf + i * config->seq_len;

        for (int j = 0; j < config->seq_len; j++) {
            float *kj = kcache + j * config->kv_hidden_size + (i / roll) * dim;
            float score = 0.0f;
            for (int k = 0; k < dim; k++) {
                score += qi[k] * kj[k];
            }
            bufi[j] = score / sqrtf(dim);
        }

        softmax(bufi, config->seq_len);

        float *oi = out + i * dim;
        memset(oi, 0, dim * sizeof(float));
        for (int j = 0; j < config->seq_len; j++) {
            float *vj = vcache + j * config->kv_hidden_size + (i / roll) * dim;
            for (int k = 0; k < dim; k++) {
                oi[k] += bufi[j] * vj[k];
            }
        }
    }

    free(buf);
}

void addres(float *out, float *in, int hidden_size) {
    for (int i = 0; i < hidden_size; i++) {
        out[i] += in[i];
    }
}

void silu(float *out, float *ffn1, float *ffn2, int hidden_size) {
    for (int i = 0; i < hidden_size; i++) {
        float val = ffn1[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= ffn2[i];
        out[i] = val;
    }
}

void llama_transformer(llama_config *config, llama_weight *weight, llama_io *lio, int pos) {
    for (int i = 0; i < config->layers; i++) {
        int offset = i * config->hidden_size;

        rmsnorm(lio->rms1, lio->token, weight->rms1 + offset, config->hidden_size);

        float *wq = weight->q + offset * config->hidden_size;
        float *wk = weight->k + offset * config->kv_hidden_size;
        float *wv = weight->v + offset * config->kv_hidden_size;
        float *kcache = lio->kcache + offset * config->seq_len + pos * config->kv_hidden_size;
        float *vcache = lio->vcache + offset * config->seq_len + pos * config->kv_hidden_size;
        matmul(lio->q, lio->rms1, wq, 1, config->hidden_size, config->hidden_size);
        matmul(kcache, lio->rms1, wk, 1, config->kv_hidden_size, config->hidden_size);
        matmul(vcache, lio->rms1, wv, 1, config->kv_hidden_size, config->hidden_size);

        // TODO rotary embedding

        attention(lio->atto, lio->q, kcache, vcache, config);

        float *wproj = weight->proj + offset * config->hidden_size;
        matmul(lio->proj, lio->atto, wproj, 1, config->hidden_size, config->hidden_size);

        addres(lio->token, lio->proj, config->hidden_size);

        rmsnorm(lio->rms2, lio->token, weight->rms2 + offset, config->hidden_size);

        float *wffn1 = weight->ffn1 + offset * config->intern_size;
        float *wffn2 = weight->ffn2 + offset * config->intern_size;
        matmul(lio->ffn1, lio->rms2, wffn1, 1, config->intern_size, config->hidden_size);
        matmul(lio->ffn2, lio->rms2, wffn2, 1, config->intern_size, config->hidden_size);

        silu(lio->silu, lio->ffn1, lio->ffn2, config->intern_size);

        float *wffn3 = weight->ffn3 + offset * config->intern_size;
        matmul(lio->ffn3, lio->silu, wffn3, 1, config->hidden_size, config->intern_size);

        addres(lio->token, lio->ffn3, config->hidden_size);
    }

    rmsnorm(lio->rms3, lio->token, weight->rms3, config->hidden_size);

    matmul(lio->logits, lio->rms3, weight->logits, 1, config->vocab_size, config->hidden_size);
}

int main(int argc, char *argv[]) {
    // TODO init

    // TODO transformer
  
    return 0;
}
