package com.sun1423.playtest

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import java.util.ArrayDeque
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

data class StreamPlaybackStats(
    val audioSeconds: Double,
    val firstChunkLatencyMs: Long,
    val chunkCount: Int,
    val underruns: Int
)

class NativeBridge {
    @Volatile
    private var libraryLoaded = false

    private var audioTrack: AudioTrack? = null
    private val pendingChunks = ArrayDeque<FloatArray>()
    private var pendingFrames = 0L
    private var requestStartedAtMs = 0L
    private var firstChunkLatencyMs = -1L
    private var framesWritten = 0L
    private var chunkCount = 0
    private var playbackStarted = false

    private val prebufferFrames = 24000L * 2L

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
    fun beginStreamingRequest() {
        check(audioTrack == null && pendingChunks.isEmpty()) {
            "A previous audio stream has not been released."
        }
        requestStartedAtMs = SystemClock.elapsedRealtime()
        firstChunkLatencyMs = -1L
        framesWritten = 0L
        pendingFrames = 0L
        chunkCount = 0
        playbackStarted = false
    }

    // Called synchronously by QVAC/GGML through JNI for each generated PCM chunk.
    // The first two seconds are prebuffered; subsequent writes apply backpressure.
    @Synchronized
    fun onPcmChunk(samples: FloatArray, isLast: Boolean) {
        require(samples.isNotEmpty()) { "QVAC delivered an empty audio chunk." }

        framesWritten += samples.size
        chunkCount++

        val currentTrack = audioTrack
        if (currentTrack == null) {
            pendingChunks.addLast(samples)
            pendingFrames += samples.size
            if (pendingFrames >= prebufferFrames || isLast) {
                startPlaybackFromPending()
            }
        } else {
            writeFully(currentTrack, samples)
        }
    }

    @Synchronized
    private fun startPlaybackFromPending() {
        check(audioTrack == null) { "Audio playback has already started." }
        check(pendingChunks.isNotEmpty()) { "No generated audio is available to start playback." }

        val track = createAudioTrack()
        audioTrack = track
        try {
            while (pendingChunks.isNotEmpty()) {
                writeFully(track, pendingChunks.removeFirst())
            }
            pendingFrames = 0L
            firstChunkLatencyMs = SystemClock.elapsedRealtime() - requestStartedAtMs
            track.play()
            playbackStarted = true
        } catch (error: Exception) {
            audioTrack = null
            pendingChunks.clear()
            pendingFrames = 0L
            track.release()
            throw error
        }
    }

    private fun writeFully(track: AudioTrack, samples: FloatArray) {
        var offset = 0
        while (offset < samples.size) {
            val written = track.write(
                samples,
                offset,
                samples.size - offset,
                AudioTrack.WRITE_BLOCKING
            )
            check(written > 0) { "AudioTrack write failed: " + written }
            offset += written
        }
    }

    suspend fun finishStreamingPlayback(): StreamPlaybackStats = withContext(Dispatchers.IO) {
        val snapshot = synchronized(this@NativeBridge) {
            if (audioTrack == null && pendingChunks.isNotEmpty()) {
                startPlaybackFromPending()
            }
            val track = audioTrack ?: throw IllegalStateException("QVAC did not deliver an audio chunk.")
            Pair(track, framesWritten)
        }

        val track = snapshot.first
        val expectedFrames = snapshot.second
        val deadline = SystemClock.elapsedRealtime() +
            (expectedFrames * 1000L / 24000L) + 15000L

        while ((track.playbackHeadPosition.toLong() and 0xffffffffL) < expectedFrames) {
            if (SystemClock.elapsedRealtime() > deadline) {
                throw IllegalStateException("Timed out waiting for Android audio playback to drain.")
            }
            delay(20)
        }

        val stats = StreamPlaybackStats(
            audioSeconds = expectedFrames.toDouble() / 24000.0,
            firstChunkLatencyMs = synchronized(this@NativeBridge) { firstChunkLatencyMs },
            chunkCount = synchronized(this@NativeBridge) { chunkCount },
            underruns = track.underrunCount
        )

        synchronized(this@NativeBridge) {
            releaseStreamingTrack()
        }
        stats
    }

    @Synchronized
    fun abortStreamingPlayback() {
        releaseStreamingTrack()
    }

    @Synchronized
    private fun releaseStreamingTrack() {
        val track = audioTrack
        audioTrack = null
        pendingChunks.clear()
        pendingFrames = 0L
        playbackStarted = false
        framesWritten = 0L
        chunkCount = 0
        if (track != null) {
            try {
                if (track.state == AudioTrack.STATE_INITIALIZED) {
                    track.pause()
                    track.flush()
                }
            } finally {
                track.release()
            }
        }
    }

    private fun createAudioTrack(): AudioTrack {
        val sampleRate = 24000
        val channel = AudioFormat.CHANNEL_OUT_MONO
        val encoding = AudioFormat.ENCODING_PCM_FLOAT
        val minimum = AudioTrack.getMinBufferSize(sampleRate, channel, encoding)
        require(minimum > 0) { "Android does not support 24 kHz float PCM playback." }

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(channel)
                    .setEncoding(encoding)
                    .build()
            )
            // Three seconds of native player buffer; start only after about two seconds are ready.
            .setBufferSizeInBytes(maxOf(minimum, sampleRate * 4 * 3))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        if (track.state != AudioTrack.STATE_INITIALIZED) {
            track.release()
            throw IllegalStateException("Android could not initialize AudioTrack.")
        }
        return track
    }

    @Synchronized
    fun synthesizeStreaming(text: String) {
        ensureLibraryLoaded()
        if (!nativeSynthesize(text)) {
            throw IllegalStateException(nativeLastError())
        }
    }

    @Synchronized
    fun unload() {
        ensureLibraryLoaded()
        releaseStreamingTrack()
        nativeDestroy()
    }

    private external fun nativeCreate(
        t3Path: String,
        s3genPath: String,
        voiceDirectory: String,
        language: String
    ): String

    private external fun nativeSynthesize(text: String): Boolean
    private external fun nativeLastError(): String
    private external fun nativeDestroy()
}
