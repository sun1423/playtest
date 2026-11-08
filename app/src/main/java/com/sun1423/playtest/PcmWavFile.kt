package com.sun1423.playtest

import android.util.Log
import java.io.File
import java.io.RandomAccessFile

internal data class PcmWavWriteStats(val bytesWritten: Long, val nonZeroSamples: Long, val minimumSample: Short, val maximumSample: Short, val rms: Double)

/** Writes QVAC PCM16 in bounded chunks to a standard little-endian RIFF/WAVE file. */
internal object PcmWavFile {
    private const val SAMPLE_RATE = 24000
    private const val CHANNELS = 1
    private const val BITS_PER_SAMPLE = 16
    private const val HEADER_BYTES = 44L

    fun writeFromNativePcm(bridge: NativeBridge, sampleCount: Long, destination: File): PcmWavWriteStats {
        require(sampleCount > 0L) { "No PCM samples were generated." }
        val dataBytes = Math.multiplyExact(sampleCount, 2L)
        require(dataBytes <= 0xFFFFFFFFL - HEADER_BYTES) { "Generated audio is too large for a standard WAV file." }
        var transferred = 0L
        var nonZero = 0L
        var minimum = Short.MAX_VALUE
        var maximum = Short.MIN_VALUE
        var squareSum = 0.0
        RandomAccessFile(destination, "rw").use { output ->
            output.setLength(0L)
            writeHeader(output, dataBytes)
            while (transferred < sampleCount) {
                val pcm = bridge.readPcm16(minOf(4096L, sampleCount - transferred).toInt())
                check(pcm.isNotEmpty()) { "Native PCM ended early: $transferred of $sampleCount samples." }
                check(transferred + pcm.size <= sampleCount) { "Native PCM exceeded the reported sample count." }
                val bytes = ByteArray(pcm.size * 2)
                for (index in pcm.indices) {
                    val sample = pcm[index]
                    if (sample < minimum) minimum = sample
                    if (sample > maximum) maximum = sample
                    if (sample.toInt() != 0) nonZero++
                    squareSum += sample.toDouble() * sample.toDouble()
                    val value = sample.toInt()
                    bytes[index * 2] = value.toByte()
                    bytes[index * 2 + 1] = (value shr 8).toByte()
                }
                output.write(bytes)
                transferred += pcm.size
            }
            check(transferred == sampleCount) { "PCM sample mismatch: $transferred of $sampleCount samples." }
            val rms = kotlin.math.sqrt(squareSum / sampleCount)
            Log.i("QVAC_AUDIO", "WAV PCM verified: samples=$transferred min=$minimum max=$maximum rms=$rms nonZero=$nonZero/$sampleCount")
            check(nonZero > sampleCount / 100L && rms > 100.0) { "Generated PCM is effectively silent (min=$minimum max=$maximum RMS=$rms)." }
            output.fd.sync()
        }
        val expectedLength = HEADER_BYTES + dataBytes
        check(destination.length() == expectedLength) { "WAV file size mismatch: ${destination.length()} of $expectedLength bytes." }
        return PcmWavWriteStats(expectedLength, nonZero, minimum, maximum, kotlin.math.sqrt(squareSum / sampleCount))
    }

    private fun writeHeader(output: RandomAccessFile, dataBytes: Long) {
        output.write("RIFF".toByteArray(Charsets.US_ASCII)); writeIntLe(output, 36L + dataBytes)
        output.write("WAVE".toByteArray(Charsets.US_ASCII)); output.write("fmt ".toByteArray(Charsets.US_ASCII))
        writeIntLe(output, 16L); writeShortLe(output, 1); writeShortLe(output, CHANNELS)
        writeIntLe(output, SAMPLE_RATE.toLong()); writeIntLe(output, SAMPLE_RATE.toLong() * CHANNELS * BITS_PER_SAMPLE / 8L)
        writeShortLe(output, CHANNELS * BITS_PER_SAMPLE / 8); writeShortLe(output, BITS_PER_SAMPLE)
        output.write("data".toByteArray(Charsets.US_ASCII)); writeIntLe(output, dataBytes)
    }
    private fun writeShortLe(output: RandomAccessFile, value: Int) { output.write(value and 0xFF); output.write((value ushr 8) and 0xFF) }
    private fun writeIntLe(output: RandomAccessFile, value: Long) {
        output.write((value and 0xFF).toInt()); output.write(((value ushr 8) and 0xFF).toInt())
        output.write(((value ushr 16) and 0xFF).toInt()); output.write(((value ushr 24) and 0xFF).toInt())
    }
}
