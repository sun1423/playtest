#include <jni.h>
#include <android/log.h>

#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/fit.h"

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

        if (!std::filesystem::is_regular_file(t3)) {
            throw std::runtime_error("T3 GGUF file is missing.");
        }
        if (!std::filesystem::is_regular_file(s3gen)) {
            throw std::runtime_error("S3Gen GGUF file is missing.");
        }
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
        fit_options.n_ctx = 1024;
        fit_options.kv_cache_type = "f16";
        fit_options.text_tokens = 256;
        fit_options.n_predict = 1000;
        fit_options.margin_bytes = 768ull * 1024ull * 1024ull;

        const auto fit = tts_cpp::chatterbox::fit_params(fit_options);
        if (fit.status != tts_cpp::FitStatus::Success || !fit.fits) {
            throw std::runtime_error(
                "QVAC memory preflight refused to load the models (" +
                fit.reason + ").\\n" + fit.report);
        }
        if (fit.model_variant != "chatterbox-t3-mtl") {
            throw std::runtime_error(
                "The selected T3 GGUF is not the Chatterbox multilingual model. "
                "Select the T3 MTL GGUF produced for the Azure voice profile.");
        }
        __android_log_print(
            ANDROID_LOG_INFO, "QVAC_FIT", "%s", fit.report.c_str());

        tts_cpp::chatterbox::EngineOptions options;
        options.t3_gguf_path = t3;
        options.s3gen_gguf_path = s3gen;
        options.voice_dir = voice;
        options.language = lang.empty() ? "en" : lang;
        options.n_ctx = 1024;
        options.kv_cache_type = "f16";
        options.n_threads = 4;
        options.n_gpu_layers = 99; // Try compiled Vulkan; QVAC may fall back to CPU.
        options.max_sentence_chars = 180;
        options.output_sample_rate = 24000;
        // Batch synthesis is intentional in this prototype: playback starts
        // only after the whole utterance is ready, so slow inference cannot
        // cause AudioTrack under-runs. Streaming can be re-enabled after the
        // target phone demonstrates RTF <= 1 on the selected backend.

        auto candidate = std::make_unique<tts_cpp::chatterbox::Engine>(options);
        {
            std::lock_guard<std::mutex> lock(g_engine_mutex);
            g_engine = std::move(candidate);
            g_last_error.clear();
        }
        return to_java(env, "");
    } catch (const std::exception & error) {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        g_engine.reset();
        g_last_error = error.what();
        return to_java(env, g_last_error);
    } catch (...) {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        g_engine.reset();
        g_last_error = "Unknown native QVAC initialization failure.";
        return to_java(env, g_last_error);
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
        if (text.size() > 1200) throw std::runtime_error("Prototype input limit is 300 characters.");

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
