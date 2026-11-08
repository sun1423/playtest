package com.sun1423.playtest

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.os.StatFs
import android.view.View
import android.os.SystemClock
import android.provider.OpenableColumns
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

private val requiredProfileFiles = setOf(
    "embedding.npy",
    "speaker_emb.npy",
    "prompt_feat.npy",
    "prompt_token.npy",
    "cond_prompt_speech_tokens.npy"
)

private data class PcmPlaybackStats(
    val audioSeconds: Double,
    val underruns: Int
)

class MainActivity : ComponentActivity() {
    private val bridge = NativeBridge()

    private lateinit var statusView: TextView
    private lateinit var textInput: EditText
    private lateinit var languagePicker: Spinner
    private lateinit var loadButton: Button
    private lateinit var playButton: Button
    private lateinit var unloadButton: Button
    private lateinit var operationProgress: ProgressBar

    private var t3Ready = false
    private var s3genReady = false
    private var voiceReady = false
    private var engineLoaded = false
    private var busy = false

    private val t3File get() = File(filesDir, "models/t3-model.gguf")
    private val s3genFile get() = File(filesDir, "models/s3gen-model.gguf")
    private val voiceDirectory get() = File(filesDir, "voice-profiles/azure-qvac")

    private val t3Picker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importModel(uri, "t3-model.gguf", true)
    }

    private val s3genPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importModel(uri, "s3gen-model.gguf", false)
    }

    private val voicePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importProfile(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // The UI opens without loading JNI, GGUF models or the QVAC engine.
        t3Ready = isValidGgufFile(t3File)
        s3genReady = isValidGgufFile(s3genFile)
        voiceReady = try {
            validateVoiceProfile(voiceDirectory)
            true
        } catch (_: Exception) {
            false
        }

        val scroll = ScrollView(this)
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(36, 32, 36, 32)
        }
        scroll.addView(column)
        setContentView(scroll)

        fun addText(value: String, size: Float = 16f, bold: Boolean = false): TextView {
            return TextView(this).apply {
                text = value
                textSize = size
                if (bold) setTypeface(typeface, android.graphics.Typeface.BOLD)
                setPadding(0, 10, 0, 10)
                column.addView(this, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            }
        }

        fun addButton(label: String, action: () -> Unit): Button {
            return Button(this).apply {
                text = label
                setOnClickListener { action() }
                column.addView(this, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            }
        }

        addText("QVAC Chatterbox Playtest", 23f, true)
        addText("GGUF + QVAC/GGML. Import the voice profile created by your Azure QVAC pipeline. Speech generation runs locally on this phone.")
        addText(
            "Required profile ZIP: embedding.npy, speaker_emb.npy, prompt_feat.npy, prompt_token.npy and cond_prompt_speech_tokens.npy.",
            13f
        )

        addButton("Select T3 multilingual GGUF") {
            if (!busy) t3Picker.launch(arrayOf("*/*"))
        }
        addButton("Select S3Gen multilingual GGUF") {
            if (!busy) s3genPicker.launch(arrayOf("*/*"))
        }
        addButton("Import Azure QVAC voice profile ZIP") {
            if (!busy) voicePicker.launch(arrayOf("application/zip", "application/octet-stream", "*/*"))
        }

        operationProgress = ProgressBar(
            this, null, android.R.attr.progressBarStyleHorizontal
        ).apply {
            max = 1000
            isIndeterminate = true
            visibility = View.GONE
        }
        column.addView(
            operationProgress,
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )

        addText("Language", 15f, true)
        languagePicker = Spinner(this)
        languagePicker.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("English", "Hindi (experimental)")
        )
        column.addView(languagePicker, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        addText("Hindi is experimental: this prototype does not add Hindi-specific external text normalization. Treat Hindi output as exploratory; English is the baseline test.", 12f)
        languagePicker.isEnabled = true
        languagePicker.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            private var firstSelection = true
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
            override fun onItemSelected(
                parent: android.widget.AdapterView<*>?,
                view: android.view.View?,
                position: Int,
                id: Long
            ) {
                if (firstSelection) {
                    firstSelection = false
                    return
                }
                if (engineLoaded && !busy) {
                    busy = true
                    engineLoaded = false
                    refreshButtons()
                    lifecycleScope.launch {
                        try {
                            withContext(Dispatchers.IO) { bridge.unload() }
                            setStatus("Language changed. Load the QVAC engine again.")
                        } catch (error: Throwable) {
                            setStatus("Engine unload failed: " + (error.message ?: error.toString()))
                        } finally {
                            busy = false
                            refreshButtons()
                        }
                    }
                }
            }
        }

        textInput = EditText(this).apply {
            hint = "Text to speak (maximum 300 characters)"
            setText("Hello. This is a local QVAC Chatterbox voice test.")
            filters = arrayOf(android.text.InputFilter.LengthFilter(300))
            minLines = 3
            gravity = android.view.Gravity.TOP
        }
        column.addView(textInput, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        loadButton = addButton("Load QVAC Engine") { loadEngine() }
        playButton = addButton("Generate and Play") { generateAndPlay() }
        unloadButton = addButton("Unload Engine / Release Memory") { unloadEngine() }

        statusView = addText("", 14f)
        addText(
            "Mobile settings: T3 context 1,024; F16 KV cache; four CPU threads; attempt Vulkan with CPU fallback. This first prototype does not include OpenCL or Hexagon.",
            12f
        )

        refreshStatus()
        refreshButtons()
    }

    override fun onDestroy() {
        // Queues native destruction after any active synthesis without blocking the UI thread.
        bridge.close()
        engineLoaded = false
        super.onDestroy()
    }

    private fun importModel(uri: Uri, targetName: String, isT3: Boolean) {
        runBusy("Copying model into private app storage...") {
            withContext(Dispatchers.IO) { bridge.unload() }
            engineLoaded = false
            val imported = withContext(Dispatchers.IO) {
                copyModelIntoPrivateStorage(this@MainActivity, uri, targetName) { copied, total ->
                    runOnUiThread {
                        if (total > 0L) {
                            operationProgress.isIndeterminate = false
                            operationProgress.progress =
                                ((copied.toDouble() * operationProgress.max) / total)
                                    .toInt().coerceIn(0, operationProgress.max)
                            setStatus(
                                "Importing " + targetName + ": " +
                                    (copied * 100 / total) + "% (" +
                                    formatMiB(copied) + " / " + formatMiB(total) + " MiB)"
                            )
                        } else {
                            operationProgress.isIndeterminate = true
                            setStatus("Importing " + targetName + ": " + formatMiB(copied) + " MiB copied...")
                        }
                    }
                }
            }
            if (isT3) t3Ready = true else s3genReady = true
            setStatus("Imported " + targetName + " (" + imported.length() / (1024 * 1024) + " MB).")
        }
    }

    private fun importProfile(uri: Uri) {
        runBusy("Validating and importing Azure QVAC voice profile...") {
            withContext(Dispatchers.IO) { bridge.unload() }
            engineLoaded = false
            withContext(Dispatchers.IO) {
                importVoiceProfileZip(this@MainActivity, uri, voiceDirectory)
            }
            voiceReady = true
            setStatus("Azure QVAC voice profile imported. All five .npy tensors are ready.")
        }
    }

    private fun loadEngine() {
        if (!t3Ready || !s3genReady || !voiceReady) {
            setStatus("Select both GGUF models and import the Azure voice profile first.")
            return
        }
        runBusy("Loading QVAC T3 and S3Gen models. Nothing is loaded until you press this button.") {
            val language = if (languagePicker.selectedItemPosition == 1) "hi" else "en"
            val message = withContext(Dispatchers.IO) {
                bridge.create(
                    t3File.absolutePath,
                    s3genFile.absolutePath,
                    voiceDirectory.absolutePath,
                    language
                )
            }
            if (message.startsWith("QVAC_ERROR:")) {
                throw IllegalStateException(message.removePrefix("QVAC_ERROR:").trim())
            }
            if (message.isNotBlank() && !message.startsWith("QVAC_OK:")) {
                throw IllegalStateException(message)
            }
            engineLoaded = true
            setStatus(
                message.removePrefix("QVAC_OK:").trim().ifBlank {
                    "QVAC engine loaded."
                }
            )
        }
    }

    private fun generateAndPlay() {
        val text = textInput.text.toString().trim()
        if (text.isBlank()) {
            setStatus("Enter a short sentence first.")
            return
        }
        if (!engineLoaded) {
            setStatus("Load the QVAC engine first.")
            return
        }

        runBusy("Generating the complete utterance first to avoid generation-time playback gaps...") {
            val generationStarted = SystemClock.elapsedRealtime()
            val pcm = withContext(Dispatchers.IO) { bridge.synthesize(text) }
            val generationMs = SystemClock.elapsedRealtime() - generationStarted
            require(pcm.isNotEmpty()) { "QVAC returned no audio." }

            val playback = playPcmWithoutGenerationGaps(pcm)
            val rtf = generationMs.toDouble() / (playback.audioSeconds * 1000.0)
            setStatus(
                "Played " + String.format("%.1f", playback.audioSeconds) + " seconds. " +
                    "Generation " + generationMs + " ms (RTF " + String.format("%.2f", rtf) + "). " +
                    "AudioTrack underruns " + playback.underruns + " (0 expected)."
            )
        }
    }

    private fun unloadEngine() {
        if (!engineLoaded) return
        runBusy("Unloading native QVAC models and caches...") {
            withContext(Dispatchers.IO) { bridge.unload() }
            engineLoaded = false
            setStatus("QVAC engine unloaded. Native model and cache resources have been released.")
        }
    }

    private fun runBusy(message: String, action: suspend () -> Unit) {
        if (busy) return
        busy = true
        setStatus(message)
        if (::operationProgress.isInitialized) {
            operationProgress.progress = 0
            operationProgress.isIndeterminate = true
            operationProgress.visibility = View.VISIBLE
        }
        refreshButtons()
        lifecycleScope.launch {
            try {
                action()
            } catch (error: Exception) {
                setStatus("Operation failed: " + (error.message ?: error.toString()))
            } finally {
                busy = false
                if (::operationProgress.isInitialized) operationProgress.visibility = View.GONE
                refreshButtons()
            }
        }
    }

    private fun formatMiB(bytes: Long): String =
        String.format("%.0f", bytes.toDouble() / (1024.0 * 1024.0))

    private fun setStatus(message: String) {
        if (::statusView.isInitialized) statusView.text = message
    }

    private fun refreshStatus() {
        setStatus(
            "Files: T3 ${if (t3Ready) "ready" else "missing"}" +
                " | S3Gen ${if (s3genReady) "ready" else "missing"}" +
                " | Azure voice ${if (voiceReady) "ready" else "missing"}" +
                ". Models are not bundled in the APK and are not loaded on app open."
        )
    }

    private fun refreshButtons() {
        if (!::loadButton.isInitialized) return
        loadButton.isEnabled = !busy && !engineLoaded && t3Ready && s3genReady && voiceReady
        playButton.isEnabled = !busy && engineLoaded
        unloadButton.isEnabled = !busy && engineLoaded
        languagePicker.isEnabled = !busy
        textInput.isEnabled = !busy
    }

    // Synthesis is complete before AudioTrack starts. This prevents a slow
    // CPU/Vulkan inference backend from starving the player between chunks.
    private suspend fun playPcmWithoutGenerationGaps(samples: FloatArray): PcmPlaybackStats =
        withContext(Dispatchers.IO) {
            require(samples.isNotEmpty()) { "No audio samples to play." }

            val sampleRate = 24000
            val channelMask = AudioFormat.CHANNEL_OUT_MONO
            val encoding = AudioFormat.ENCODING_PCM_16BIT
            val minBufferBytes = AudioTrack.getMinBufferSize(sampleRate, channelMask, encoding)
            require(minBufferBytes > 0) { "Android does not support 24 kHz mono PCM16 playback." }

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
                        .setChannelMask(channelMask)
                        .setEncoding(encoding)
                        .build()
                )
                // Four seconds of buffer; prime one second before starting playback.
                .setBufferSizeInBytes(maxOf(minBufferBytes, sampleRate * 2 * 4))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            try {
                check(track.state == AudioTrack.STATE_INITIALIZED) {
                    "Android could not initialize AudioTrack."
                }

                fun writeRange(start: Int, count: Int) {
                    var sourceOffset = start
                    val end = start + count
                    while (sourceOffset < end) {
                        val chunkCount = minOf(4096, end - sourceOffset)
                        val pcm16 = ShortArray(chunkCount)
                        for (i in 0 until chunkCount) {
                            val value = samples[sourceOffset + i]
                            val safe = if (value.isFinite()) value.coerceIn(-1.0f, 1.0f) else 0.0f
                            val scaled = if (safe <= -1.0f) Short.MIN_VALUE.toInt()
                                else (safe * 32767.0f).toInt()
                            pcm16[i] = scaled.toShort()
                        }

                        var writtenOffset = 0
                        while (writtenOffset < pcm16.size) {
                            val written = track.write(
                                pcm16,
                                writtenOffset,
                                pcm16.size - writtenOffset,
                                AudioTrack.WRITE_BLOCKING
                            )
                            check(written > 0) { "AudioTrack write failed: " + written }
                            writtenOffset += written
                        }
                        sourceOffset += chunkCount
                    }
                }

                // Prime before play(); all model inference has already finished.
                val prebufferSamples = minOf(samples.size, sampleRate)
                writeRange(0, prebufferSamples)
                track.play()
                writeRange(prebufferSamples, samples.size - prebufferSamples)

                val expectedFrames = samples.size.toLong()
                val deadline = SystemClock.elapsedRealtime() +
                    (expectedFrames * 1000L / sampleRate) + 10000L
                while ((track.playbackHeadPosition.toLong() and 0xffffffffL) < expectedFrames) {
                    check(SystemClock.elapsedRealtime() <= deadline) {
                        "Timed out waiting for Android audio playback to finish."
                    }
                    delay(20)
                }

                PcmPlaybackStats(
                    audioSeconds = expectedFrames.toDouble() / sampleRate,
                    underruns = track.underrunCount
                )
            } finally {
                track.release()
            }
        }

}

private fun copyModelIntoPrivateStorage(
    context: Context,
    uri: Uri,
    targetName: String,
    onProgress: (copiedBytes: Long, totalBytes: Long) -> Unit
): File {
    val directory = File(context.filesDir, "models")
    check(directory.mkdirs() || directory.isDirectory) { "Cannot create private model directory." }

    val sourceSize = queryDocumentSize(context, uri)
    require(sourceSize <= 8L * 1024L * 1024L * 1024L) { "Selected file exceeds the 8 GB prototype limit." }
    if (sourceSize > 0L) {
        val available = StatFs(directory.absolutePath).availableBytes
        require(available > sourceSize + 64L * 1024L * 1024L) {
            "Not enough free app storage to copy this model safely."
        }
    }

    val target = File(directory, targetName)
    val temporary = File(directory, targetName + ".part")
    if (temporary.exists()) temporary.delete()

    try {
        val input = context.contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Android could not open the selected file.")
        var copied = 0L
        var nextProgressUpdate = 4L * 1024L * 1024L
        onProgress(0L, sourceSize)
        input.use { source ->
            FileOutputStream(temporary).use { output ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = source.read(buffer)
                    if (count < 0) break
                    copied += count
                    require(copied <= 8L * 1024L * 1024L * 1024L) { "Model exceeded the 8 GB import limit." }
                    output.write(buffer, 0, count)
                    if (copied >= nextProgressUpdate || (sourceSize > 0L && copied >= sourceSize)) {
                        onProgress(copied, sourceSize)
                        nextProgressUpdate = copied + 4L * 1024L * 1024L
                    }
                }
                output.flush()
                output.fd.sync()
            }
        }

        onProgress(copied, sourceSize)
        require(copied >= 1024L * 1024L) { "Selected file is too small to be a GGUF model." }
        val header = ByteArray(4)
        temporary.inputStream().use { input ->
            require(input.read(header) == 4 && header.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46))) {
                "Selected file does not have a GGUF header."
            }
        }

        val backup = File(directory, targetName + ".backup")
        if (backup.exists()) backup.delete()
        if (target.exists()) check(target.renameTo(backup)) { "Could not preserve the previous model." }
        if (!temporary.renameTo(target)) {
            if (backup.exists()) backup.renameTo(target)
            throw IllegalStateException("Could not finalize copied model file.")
        }
        backup.delete()
        return target
    } catch (error: Throwable) {
        temporary.delete()
        throw error
    }
}

private fun importVoiceProfileZip(context: Context, uri: Uri, destination: File) {
    val temporary = File(destination.parentFile, ".azure-qvac-profile-import")
    temporary.deleteRecursively()
    check(temporary.mkdirs() || temporary.isDirectory) { "Cannot create temporary profile directory." }

    val imported = mutableSetOf<String>()
    var totalBytes = 0L

    try {
        val input = context.contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Android could not open the selected ZIP.")

        input.use { raw ->
            ZipInputStream(raw).use { zip ->
                var entry = zip.nextEntry
                while (entry != null) {
                    val name = entry.name.substringAfterLast('/')
                    if (!entry.isDirectory && name in requiredProfileFiles) {
                        require(imported.add(name)) { "ZIP contains duplicate tensor: " + name }
                        val outputFile = File(temporary, name)
                        var entryBytes = 0L

                        FileOutputStream(outputFile).use { output ->
                            val buffer = ByteArray(16 * 1024)
                            while (true) {
                                val count = zip.read(buffer)
                                if (count < 0) break
                                entryBytes += count
                                totalBytes += count
                                require(entryBytes <= 20L * 1024L * 1024L) { name + " exceeds the 20 MB file limit." }
                                require(totalBytes <= 50L * 1024L * 1024L) { "Profile ZIP exceeds the 50 MB total limit." }
                                output.write(buffer, 0, count)
                            }
                        }

                        require(entryBytes >= 10L) { name + " is empty or truncated." }
                        val expectedDtype = if (name == "prompt_token.npy" ||
                            name == "cond_prompt_speech_tokens.npy") "<i4" else "<f4"
                        validateNpyTensor(
                            outputFile,
                            expectedDtype,
                            requirePromptFeatureShape = name == "prompt_feat.npy"
                        )
                    }
                    zip.closeEntry()
                    entry = zip.nextEntry
                }
            }
        }

        val missing = requiredProfileFiles - imported
        require(missing.isEmpty()) { "ZIP is missing: " + missing.joinToString() }
        validateVoiceProfile(temporary)

        val backup = File(destination.parentFile, ".azure-qvac-profile-backup")
        backup.deleteRecursively()
        if (destination.exists()) {
            check(destination.renameTo(backup)) { "Could not preserve the previous voice profile." }
        }
        if (!temporary.renameTo(destination)) {
            if (backup.exists()) backup.renameTo(destination)
            throw IllegalStateException("Could not finalize voice profile import.")
        }
        backup.deleteRecursively()
    } catch (error: Throwable) {
        temporary.deleteRecursively()
        throw error
    }
}

private fun validateVoiceProfile(directory: File) {
    require(directory.isDirectory) { "Voice profile directory is missing." }
    val specs = mapOf(
        "embedding.npy" to "<f4",
        "speaker_emb.npy" to "<f4",
        "prompt_feat.npy" to "<f4",
        "prompt_token.npy" to "<i4",
        "cond_prompt_speech_tokens.npy" to "<i4"
    )
    for ((name, dtype) in specs) {
        validateNpyTensor(
            File(directory, name),
            dtype,
            requirePromptFeatureShape = name == "prompt_feat.npy"
        )
    }
}

private fun validateNpyTensor(
    file: File,
    expectedDtype: String,
    requirePromptFeatureShape: Boolean
) {
    require(file.isFile && file.length() <= 20L * 1024L * 1024L) {
        "Missing or oversized voice tensor: " + file.name
    }

    java.io.DataInputStream(file.inputStream().buffered()).use { input ->
        val magic = ByteArray(6)
        input.readFully(magic)
        require(magic.contentEquals(byteArrayOf(
            0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(),
            'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()
        ))) { file.name + " is not a NumPy .npy tensor." }

        val major = input.readUnsignedByte()
        input.readUnsignedByte() // minor version
        require(major in 1..3) { file.name + " uses an unsupported NumPy format version." }

        val headerLengthBytes = if (major == 1) 2 else 4
        val headerLength: Long = if (major == 1) {
            (input.readUnsignedByte() or (input.readUnsignedByte() shl 8)).toLong()
        } else {
            input.readUnsignedByte().toLong() or
                (input.readUnsignedByte().toLong() shl 8) or
                (input.readUnsignedByte().toLong() shl 16) or
                (input.readUnsignedByte().toLong() shl 24)
        }

        require(headerLength in 1L..65536L) { file.name + " has an invalid NumPy header length." }
        val headerBytes = ByteArray(headerLength.toInt())
        input.readFully(headerBytes)
        val header = headerBytes.toString(Charsets.US_ASCII)

        val dtype = Regex("""['"]descr['"]\s*:\s*['"]([^'"]+)['"]""")
            .find(header)?.groupValues?.get(1)
            ?: throw IllegalArgumentException(file.name + " is missing its NumPy dtype.")
        require(dtype == expectedDtype) {
            file.name + " has dtype " + dtype + "; expected " + expectedDtype + "."
        }

        val fortranOrder = Regex("""['"]fortran_order['"]\s*:\s*(True|False)""")
            .find(header)?.groupValues?.get(1)
            ?: throw IllegalArgumentException(file.name + " is missing its NumPy storage-order flag.")
        require(fortranOrder == "False") { file.name + " must use contiguous C-order storage." }

        val shapeText = Regex("""['"]shape['"]\s*:\s*\(([^)]*)\)""")
            .find(header)?.groupValues?.get(1)
            ?: throw IllegalArgumentException(file.name + " is missing its NumPy shape.")

        val dimensions = shapeText.split(',')
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map { it.toLongOrNull() ?: throw IllegalArgumentException(file.name + " has an invalid shape.") }

        require(dimensions.isNotEmpty() && dimensions.all { it > 0L }) {
            file.name + " contains an empty or invalid tensor shape."
        }
        if (requirePromptFeatureShape) {
            require(dimensions.size == 2 && dimensions[1] == 80L) {
                "prompt_feat.npy must have shape (frames, 80)."
            }
        }

        val elements = dimensions.fold(1L) { total, dimension ->
            Math.multiplyExact(total, dimension)
        }
        val payloadBytes = Math.multiplyExact(elements, 4L)
        val headerPrefixBytes = 6L + 2L + headerLengthBytes.toLong()
        val expectedFileBytes = Math.addExact(
            Math.addExact(headerPrefixBytes, headerLength),
            payloadBytes
        )
        require(file.length() == expectedFileBytes) {
            file.name + " has a truncated or inconsistent NumPy payload."
        }
    }
}

private fun isValidGgufFile(file: File): Boolean {
    if (!file.isFile || file.length() < 1024L * 1024L) return false
    return try {
        val magic = ByteArray(4)
        file.inputStream().use { it.read(magic) == 4 && magic.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46)) }
    } catch (_: Exception) {
        false
    }
}

private fun queryDocumentSize(context: Context, uri: Uri): Long {
    return try {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getLong(0) else -1L
        } ?: -1L
    } catch (_: Exception) {
        -1L
    }
}
