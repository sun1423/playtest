package com.sun1423.playtest

import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException

/**
 * One process-wide native worker. QVAC's Engine is created, used, and destroyed
 * on the same thread, including across Activity recreation.
 */
private object QvacNativeWorker {
    @Volatile
    var libraryLoaded = false
        private set

    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "qvac-native-engine").apply {
            isDaemon = false
            priority = Thread.NORM_PRIORITY
        }
    }

    fun ensureLibraryLoaded() {
        if (!libraryLoaded) {
            System.loadLibrary("qvac_tts_jni")
            libraryLoaded = true
        }
    }

    fun <T> call(action: () -> T): T {
        try {
            return executor.submit<T> { action() }.get()
        } catch (error: ExecutionException) {
            val cause = error.cause
            if (cause is RuntimeException) throw cause
            throw IllegalStateException("QVAC native worker failed.", cause ?: error)
        } catch (error: InterruptedException) {
            Thread.currentThread().interrupt()
            throw IllegalStateException("Interrupted while waiting for the QVAC native worker.", error)
        } catch (error: RejectedExecutionException) {
            throw IllegalStateException("QVAC native worker is unavailable.", error)
        }
    }

    fun enqueue(action: () -> Unit) {
        try {
            executor.execute(action)
        } catch (_: RejectedExecutionException) {
            // The process is already shutting down.
        }
    }
}

class NativeBridge {
    fun create(t3Path: String, s3genPath: String, voiceDirectory: String, language: String): String =
        QvacNativeWorker.call {
            QvacNativeWorker.ensureLibraryLoaded()
            nativeCreate(t3Path, s3genPath, voiceDirectory, language)
        }

    fun synthesize(text: String): Long = QvacNativeWorker.call {
        QvacNativeWorker.ensureLibraryLoaded()
        nativeSynthesize(text).also {
            if (it <= 0L) throw IllegalStateException(nativeLastError())
        }
    }

    fun readPcm16(maxSamples: Int = 4096): ShortArray = QvacNativeWorker.call {
        QvacNativeWorker.ensureLibraryLoaded()
        nativeReadPcm16(maxSamples) ?: throw IllegalStateException(nativeLastError())
    }

    fun releasePcm() = QvacNativeWorker.call {
        if (QvacNativeWorker.libraryLoaded) nativeReleasePcm()
    }

    fun unload() = QvacNativeWorker.call {
        if (QvacNativeWorker.libraryLoaded) nativeDestroy()
    }

    /**
     * Queue cleanup behind any active native call, without blocking Android's
     * UI thread. The shared worker remains alive for a recreated Activity.
     */
    fun close() {
        QvacNativeWorker.enqueue {
            if (QvacNativeWorker.libraryLoaded) nativeDestroy()
        }
    }

    private external fun nativeCreate(
        t3Path: String,
        s3genPath: String,
        voiceDirectory: String,
        language: String
    ): String

    private external fun nativeSynthesize(text: String): Long
    private external fun nativeReadPcm16(maxSamples: Int): ShortArray?
    private external fun nativeReleasePcm()
    private external fun nativeLastError(): String
    private external fun nativeDestroy()
}
