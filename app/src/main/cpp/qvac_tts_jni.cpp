#include <jni.h>

#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "tts-cpp/chatterbox/engine.h"

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
        options.stream_chunk_tokens = 25;
        options.stream_first_chunk_tokens = 10;
        options.stream_left_context_tokens = 25;

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

extern "C" JNIEXPORT jboolean JNICALL
Java_com_sun1423_playtest_NativeBridge_nativeSynthesize(
    JNIEnv * env, jobject bridge, jstring text_value) {

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    jclass bridge_class = nullptr;

    try {
        if (!g_engine) throw std::runtime_error("QVAC engine is not loaded.");

        const std::string text = from_java(env, text_value);
        if (text.empty()) throw std::runtime_error("Enter a short sentence first.");
        if (text.size() > 1200) throw std::runtime_error("Prototype input limit is 300 characters.");

        bridge_class = env->GetObjectClass(bridge);
        if (bridge_class == nullptr) throw std::runtime_error("Could not resolve Android audio callback class.");

        const jmethodID on_chunk = env->GetMethodID(bridge_class, "onPcmChunk", "([F)V");
        if (on_chunk == nullptr) {
            if (env->ExceptionCheck()) env->ExceptionClear();
            throw std::runtime_error("Android PCM chunk callback is unavailable.");
        }

        const auto result = g_engine->synthesize(
            text,
            [env, bridge, on_chunk](const float * pcm, std::size_t samples, int, bool) {
                if (samples == 0) return;
                if (samples > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
                    throw std::runtime_error("QVAC produced an audio chunk that is too large.");
                }

                jfloatArray chunk = env->NewFloatArray(static_cast<jsize>(samples));
                if (chunk == nullptr) {
                    if (env->ExceptionCheck()) env->ExceptionClear();
                    throw std::runtime_error("Could not allocate an Android audio chunk.");
                }

                env->SetFloatArrayRegion(
                    chunk, 0, static_cast<jsize>(samples), pcm);

                if (!env->ExceptionCheck()) {
                    env->CallVoidMethod(bridge, on_chunk, chunk);
                }

                const bool callback_failed = env->ExceptionCheck();
                env->DeleteLocalRef(chunk);

                if (callback_failed) {
                    env->ExceptionClear();
                    throw std::runtime_error("Android AudioTrack rejected a QVAC audio chunk.");
                }
            });

        if (result.pcm.empty()) {
            throw std::runtime_error("QVAC completed without producing audio.");
        }

        env->DeleteLocalRef(bridge_class);
        g_last_error.clear();
        return JNI_TRUE;
    } catch (const std::exception & error) {
        if (bridge_class != nullptr) env->DeleteLocalRef(bridge_class);
        g_last_error = error.what();
        return JNI_FALSE;
    } catch (...) {
        if (bridge_class != nullptr) env->DeleteLocalRef(bridge_class);
        g_last_error = "Unknown native QVAC streaming failure.";
        return JNI_FALSE;
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
