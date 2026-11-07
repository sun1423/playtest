package com.sun1423.playtest

import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException

/**
 * Owns one native worker thread for the full QVAC Engine lifetime.
 *
 * The Engine is created, used, and destroyed on this same thread. This avoids
 * moving ggml thread-local graph/cache state between Dispatchers.IO workers.
 */
class NativeBridge {
    @Volatile
    private var libraryLoaded = false

    private val nativeExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "qvac-native-engine").apply {
            isDaemon = false
            priority = Thread.NORM_PRIORITY
        }
    }

    private fun <T> runOnNativeThread(action: () -> T): T {
        try {
            return nativeExecutor.submit<T> { action() }.get()
        } catch (error: ExecutionException) {
            val cause = error.cause
            if (cause is RuntimeException) throw cause
            throw IllegalStateException("QVAC native worker failed.", cause ?: error)
        } catch (error: InterruptedException) {
            Thread.currentThread().interrupt()
            throw IllegalStateException("Interrupted while waiting for the QVAC native worker.", error)
        } catch (error: RejectedExecutionException) {
            throw IllegalStateException("QVAC native worker has already been closed.", error)
        }
    }

    private fun ensureLibraryLoaded() {
        if (!libraryLoaded) {
            System.loadLibrary("qvac_tts_jni")
            libraryLoaded = true
        }
    }

    fun create(t3Path: String, s3genPath: String, voiceDirectory: String, language: String): String =
        runOnNativeThread {
            ensureLibraryLoaded()
            nativeCreate(t3Path, s3genPath, voiceDirectory, language)
        }

    fun synthesize(text: String): FloatArray = runOnNativeThread {
        ensureLibraryLoaded()
        nativeSynthesize(text) ?: throw IllegalStateException(nativeLastError())
    }

    fun unload() = runOnNativeThread {
        if (libraryLoaded) nativeDestroy()
    }

    /**
     * Queue destruction after any in-flight create/synthesis call, without
     * blocking Android's UI thread during Activity teardown.
     */
    fun close() {
        try {
            nativeExecutor.execute {
                try {
                    if (libraryLoaded) nativeDestroy()
                } catch (_: Exception) {
                    // The Android Activity is already closing; native process
                    // teardown remains the final fallback if a driver refuses release.
                }
            }
        } catch (_: RejectedExecutionException) {
            // Already closed.
        } finally {
            nativeExecutor.shutdown()
        }
    }

    private external fun nativeCreate(
        t3Path: String,
        s3genPath: String,
        voiceDirectory: String,
        language: String
    ): String

    private external fun nativeSynthesize(text: String): FloatArray?
    private external fun nativeLastError(): String
    private external fun nativeDestroy()
}
