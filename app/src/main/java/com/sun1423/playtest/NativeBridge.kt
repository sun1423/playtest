package com.sun1423.playtest

class NativeBridge {
    @Volatile
    private var libraryLoaded = false

    @Synchronized
    private fun ensureLibraryLoaded() {
        if (!libraryLoaded) {
            System.loadLibrary("qvac_tts_jni")
            libraryLoaded = true
        }
    }

    @Synchronized
    fun create(t3Path: String, s3genPath: String, voiceDirectory: String, language: String): String {
        ensureLibraryLoaded()
        return nativeCreate(t3Path, s3genPath, voiceDirectory, language)
    }

    @Synchronized
    fun synthesize(text: String): FloatArray {
        ensureLibraryLoaded()
        return nativeSynthesize(text) ?: throw IllegalStateException(nativeLastError())
    }

    @Synchronized
    fun unload() {
        if (libraryLoaded) nativeDestroy()
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
