#include <jni.h>
#include <android/log.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/fit.h"
#include "ggml.h"
#include "gguf.h"

namespace {
std::mutex g_engine_mutex;
std::unique_ptr<tts_cpp::chatterbox::Engine> g_engine;
std::string g_last_error;

std::string from_java(JNIEnv * env, jstring value) {
    if (value == nullptr) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jstring to_java(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

std::string read_gguf_string(const std::string & path, const char * key) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    params.kv_only = true;

    std::unique_ptr<gguf_context, void (*)(gguf_context *)> context(
        gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!context) {
        throw std::runtime_error(std::string("Cannot read GGUF metadata: ") + path);
    }

    const int64_t key_id = gguf_find_key(context.get(), key);
    if (key_id < 0 || gguf_get_kv_type(context.get(), key_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("GGUF metadata key is missing or not a string: ") + key);
    }

    const char * value = gguf_get_val_str(context.get(), key_id);
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("GGUF metadata value is empty: ") + key);
    }
    return value;
}

struct QuantizationSummary {
    int q4_0_tensors = 0;
    int q8_0_tensors = 0;
    uint64_t q4_0_bytes = 0;
    uint64_t q8_0_bytes = 0;
};

QuantizationSummary read_quantization_summary(const std::string & path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    params.kv_only = false;

    std::unique_ptr<gguf_context, void (*)(gguf_context *)> context(
        gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!context) {
        throw std::runtime_error("Cannot inspect GGUF tensor types: " + path);
    }

    QuantizationSummary summary;
    const int64_t tensor_count = gguf_get_n_tensors(context.get());
    for (int64_t i = 0; i < tensor_count; ++i) {
        const auto type = gguf_get_tensor_type(context.get(), i);
        const auto bytes = static_cast<uint64_t>(gguf_get_tensor_size(context.get(), i));
        if (type == GGML_TYPE_Q4_0) {
            ++summary.q4_0_tensors;
            summary.q4_0_bytes += bytes;
        } else if (type == GGML_TYPE_Q8_0) {
            ++summary.q8_0_tensors;
            summary.q8_0_bytes += bytes;
        }
    }
    return summary;
}

void require_q4_0_model(
    const std::string & path,
    const char * label,
    int minimum_q4_tensors) {
    const auto summary = read_quantization_summary(path);
    if (summary.q4_0_tensors < minimum_q4_tensors || summary.q8_0_tensors != 0) {
        throw std::runtime_error(
            std::string(label) + " model must be the Q4_0 file. Detected " +
            std::to_string(summary.q4_0_tensors) + " Q4_0 tensors (" +
            std::to_string(summary.q4_0_bytes / (1024 * 1024)) + " MiB) and " +
            std::to_string(summary.q8_0_tensors) + " Q8_0 tensors (" +
            std::to_string(summary.q8_0_bytes / (1024 * 1024)) +
            " MiB). Re-import the matching Q4_0 multilingual model.");
    }
}

void require_profile(const std::string & directory) {
    static const std::array<const char *, 5> required = {
        "embedding.npy",
        "speaker_emb.npy",
        "prompt_feat.npy",
        "prompt_token.npy",
        "cond_prompt_speech_tokens.npy"
    };

    for (const char * name : required) {
        const auto path = std::filesystem::path(directory) / name;
        if (!std::filesystem::is_regular_file(path) ||
            std::filesystem::file_size(path) < 16) {
            throw std::runtime_error(std::string("Voice profile is missing or invalid: ") + name);
        }
    }
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_sun1423_playtest_NativeBridge_nativeCreate(
    JNIEnv * env, jobject,
    jstring t3_path,
    jstring s3gen_path,
    jstring voice_directory,
    jstring language) {

    try {
        const std::string t3 = from_java(env, t3_path);
        const std::string s3gen = from_java(env, s3gen_path);
        const std::string voice = from_java(env, voice_directory);
        const std::string lang = from_java(env, language);

        // Enable sequential S3Gen stage-cache reclamation before either the
        // metadata preflight or the real Engine is constructed.
        if (setenv("TTS_CPP_CHATTERBOX_LOW_MEMORY", "1", 1) != 0) {
            throw std::runtime_error("Could not enable the QVAC low-memory runtime mode.");
        }

        if (!std::filesystem::is_regular_file(t3)) {
            throw std::runtime_error("T3 GGUF file is missing.");
        }
        if (!std::filesystem::is_regular_file(s3gen)) {
            throw std::runtime_error("S3Gen GGUF file is missing.");
        }

        const std::string t3_variant = read_gguf_string(t3, "chatterbox.variant");
        const std::string s3_variant = read_gguf_string(s3gen, "s3gen.variant");
        if (t3_variant != "t3_mtl" || s3_variant != "mtl") {
            throw std::runtime_error(
                "Model mismatch: this prototype requires T3 variant t3_mtl and S3Gen variant mtl; "
                "found T3='" + t3_variant + "', S3Gen='" + s3_variant + "'.");
        }

        // Do not accept Q8 imports on a phone profile tuned for Q4_0.
        // S3Gen intentionally keeps many F32 tensors; validate its distinctive
        // Q4_0 tensor inventory rather than using a crude file-size threshold.
        require_q4_0_model(t3, "T3", 200);
        require_q4_0_model(s3gen, "S3Gen", 300);
        require_profile(voice);

        // Release an old engine before constructing another one to avoid holding
        // two multi-hundred-megabyte models during a language/profile change.
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            g_engine.reset();
            g_last_error.clear();
        }

        tts_cpp::chatterbox::FitOptions fit_options;
        fit_options.t3_gguf_path = t3;
        fit_options.s3gen_gguf_path = s3gen;
        fit_options.n_gpu_layers = 99;
        fit_options.n_ctx = 320;
        fit_options.kv_cache_type = "f16";
        fit_options.text_tokens = 192; // conservative ceiling for the 120-character input field
        fit_options.low_memory = true;
        fit_options.margin_bytes = 768ull * 1024ull * 1024ull;

        // Keep QVAC's 768 MiB safety margin. Add 128 MiB headroom on
        // sub-6-GiB devices and 256 MiB on larger devices; this adapts to
        // Android's smaller memory pool without removing the QVAC reserve.
        constexpr uint64_t SMALL_DEVICE_EXTRA_HEADROOM = 128ull * 1024ull * 1024ull;
        constexpr uint64_t LARGE_DEVICE_EXTRA_HEADROOM = 256ull * 1024ull * 1024ull;
        constexpr uint64_t SMALL_DEVICE_RAM_LIMIT = 6ull * 1024ull * 1024ull * 1024ull;
        constexpr std::array<int, 3> SPEECH_TOKEN_BUDGETS = {80, 64, 48};
        tts_cpp::FitResult selected_fit;
        int selected_budget = 0;

        for (const int budget : SPEECH_TOKEN_BUDGETS) {
            fit_options.n_predict = budget;
            const auto attempt = tts_cpp::chatterbox::fit_params(fit_options);
            selected_fit = attempt;

            if (attempt.status == tts_cpp::FitStatus::Error) {
                throw std::runtime_error(
                    "QVAC memory preflight failed (" + attempt.reason + ").\\n" +
                    attempt.report);
            }
            if (attempt.status != tts_cpp::FitStatus::Success || !attempt.fits) {
                continue;
            }

            uint64_t required = attempt.device.total_bytes + fit_options.margin_bytes;
            if (attempt.device_shares_host_memory) {
                required += attempt.host_bytes;
            }
            const uint64_t extra_headroom =
                attempt.device_total_bytes < SMALL_DEVICE_RAM_LIMIT
                    ? SMALL_DEVICE_EXTRA_HEADROOM
                    : LARGE_DEVICE_EXTRA_HEADROOM;
            if (attempt.device_free_bytes >= required &&
                attempt.device_free_bytes - required >= extra_headroom) {
                selected_budget = budget;
                break;
            }
        }

        // A failed fit estimate is a warning, not an automatic load veto.
        // The estimator is conservative and cannot know every allocation the
        // Android runtime/backend will actually make. Try the smallest workload
        // once; the real Engine constructor remains the authority on whether
        // model/backend initialization succeeds. Never weaken model validation
        // or silently change Q4/F16/context settings to manufacture a fit.
        const bool memory_preflight_warning = selected_budget == 0;
        if (memory_preflight_warning) {
            selected_budget = 48;
            fit_options.n_predict = selected_budget;
            __android_log_print(
                ANDROID_LOG_WARN, "QVAC_FIT",
                "Memory preflight did not pass; continuing with the minimum "
                "48-token workload. Runtime allocation may still fail.\\n%s",
                selected_fit.report.c_str());
        }
        if (selected_fit.model_variant != "chatterbox-t3-mtl") {
            throw std::runtime_error(
                "The selected T3 GGUF is not the Chatterbox multilingual model. "
                "Select the T3 MTL GGUF produced for the Azure voice profile.");
        }
        __android_log_print(
            ANDROID_LOG_INFO, "QVAC_FIT",
            "Selected speech budget: %d tokens\\n%s",
            selected_budget, selected_fit.report.c_str());

        tts_cpp::chatterbox::EngineOptions options;
        options.t3_gguf_path = t3;
        options.s3gen_gguf_path = s3gen;
        options.voice_dir = voice;
        options.language = lang.empty() ? "en" : lang;
        options.n_ctx = 320;
        options.n_predict = selected_budget;
        options.kv_cache_type = "f16";
        // Keep CPU scratch/thread pressure lower on smaller-RAM phones.
        options.n_threads = selected_fit.device_total_bytes < (6ull * 1024ull * 1024ull * 1024ull) ? 2 : 4;
        options.cfm_steps = 7; // Multilingual quality knee: fewer S3Gen steps, lower latency.
        options.n_gpu_layers = 99; // Vulkan when available; QVAC safely falls back to CPU.
        options.verbose = true; // Surface precise GPU discovery / initialization diagnostics.
        options.max_sentence_chars = 480; // 120 UTF-8 characters without splitting Hindi code points.
        options.output_sample_rate = 24000;
        // Batch synthesis is intentional in this prototype: playback starts
        // only after the whole utterance is ready, so slow inference cannot
        // cause AudioTrack under-runs. Streaming can be re-enabled after the
        // target phone demonstrates RTF <= 1 on the selected backend.

        auto candidate = std::make_unique<tts_cpp::chatterbox::Engine>(options);
        const std::string backend_name = candidate->backend_name();
        const bool gpu_unvalidated = candidate->gpu_unsupported();
        __android_log_print(
            ANDROID_LOG_INFO, "QVAC_RUNTIME",
            "Backend=%s; GPU skipped by validation policy=%s; n_ctx=%d; n_predict=%d; kv=f16; low_memory=1",
            backend_name.c_str(), gpu_unvalidated ? "true" : "false",
            options.n_ctx, options.n_predict);
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            g_engine = std::move(candidate);
            g_last_error.clear();
        }
        return to_java(env,
            "QVAC_OK: Loaded on " + backend_name +
            " with verified Q4_0 models, low-memory mode, F16 KV, context 320 and a " +
            std::to_string(selected_budget) + "-speech-token limit." +
            (memory_preflight_warning
                ? " WARNING: Memory preflight was below the recommended reserve; "
                  "the minimum workload was attempted. Synthesis may still fail "
                  "if Android cannot provide the required memory."
                : ""));
    } catch (const std::exception & error) {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        g_engine.reset();
        g_last_error = error.what();
        return to_java(env, "QVAC_ERROR: " + g_last_error);
    } catch (...) {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        g_engine.reset();
        g_last_error = "Unknown native QVAC initialization failure.";
        return to_java(env, "QVAC_ERROR: " + g_last_error);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_sun1423_playtest_NativeBridge_nativeSynthesize(
    JNIEnv * env, jobject, jstring text_value) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    try {
        if (!g_engine) throw std::runtime_error("QVAC engine is not loaded.");

        const std::string text = from_java(env, text_value);
        if (text.empty()) throw std::runtime_error("Enter a short sentence first.");
        if (text.size() > 1200) throw std::runtime_error("Prototype input limit is 120 characters (UTF-8 byte guard).");

        const auto result = g_engine->synthesize(text);
        if (result.pcm.empty()) {
            throw std::runtime_error("QVAC completed without producing audio.");
        }
        if (result.pcm.size() > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
            throw std::runtime_error("Generated audio is too large for the Android playback buffer.");
        }

        jfloatArray output = env->NewFloatArray(static_cast<jsize>(result.pcm.size()));
        if (output == nullptr) {
            if (env->ExceptionCheck()) env->ExceptionClear();
            throw std::runtime_error("Android could not allocate the generated PCM buffer.");
        }

        env->SetFloatArrayRegion(
            output, 0, static_cast<jsize>(result.pcm.size()), result.pcm.data());
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            env->DeleteLocalRef(output);
            throw std::runtime_error("Android could not copy generated PCM.");
        }

        g_last_error.clear();
        return output;
    } catch (const std::exception & error) {
        g_last_error = error.what();
        return nullptr;
    } catch (...) {
        g_last_error = "Unknown native QVAC synthesis failure.";
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_sun1423_playtest_NativeBridge_nativeLastError(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    return to_java(env, g_last_error);
}

extern "C" JNIEXPORT void JNICALL
Java_com_sun1423_playtest_NativeBridge_nativeDestroy(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    g_engine.reset();
    g_last_error.clear();
}
