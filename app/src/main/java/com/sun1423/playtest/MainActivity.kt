package com.sun1423.playtest

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.os.StatFs
import android.provider.OpenableColumns
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
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

class MainActivity : ComponentActivity() {
    private val bridge = NativeBridge()

    private lateinit var statusView: TextView
    private lateinit var textInput: EditText
    private lateinit var languagePicker: Spinner
    private lateinit var loadButton: Button
    private lateinit var playButton: Button
    private lateinit var unloadButton: Button

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
        t3Ready = t3File.isFile && t3File.length() > 1024 * 1024
        s3genReady = s3genFile.isFile && s3genFile.length() > 1024 * 1024
        voiceReady = requiredProfileFiles.all { File(voiceDirectory, it).isFile }

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

        addText("Language", 15f, true)
        languagePicker = Spinner(this)
        languagePicker.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("English", "Hindi")
        )
        column.addView(languagePicker, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
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
                    engineLoaded = false
                    lifecycleScope.launch(Dispatchers.IO) { bridge.unload() }
                    statusView.text = "Language changed. Load the QVAC engine again."
                    refreshButtons()
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

    private fun importModel(uri: Uri, targetName: String, isT3: Boolean) {
        runBusy("Copying model into private app storage...") {
            val imported = withContext(Dispatchers.IO) {
                copyModelIntoPrivateStorage(this@MainActivity, uri, targetName)
            }
            withContext(Dispatchers.IO) { bridge.unload() }
            engineLoaded = false
            if (isT3) t3Ready = true else s3genReady = true
            setStatus("Imported " + targetName + " (" + imported.length() / (1024 * 1024) + " MB).")
        }
    }

    private fun importProfile(uri: Uri) {
        runBusy("Validating and importing Azure QVAC voice profile...") {
            withContext(Dispatchers.IO) {
                importVoiceProfileZip(this@MainActivity, uri, voiceDirectory)
            }
            withContext(Dispatchers.IO) { bridge.unload() }
            engineLoaded = false
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
            if (message.isNotBlank()) throw IllegalStateException(message)
            engineLoaded = true
            setStatus("QVAC engine loaded. Vulkan is attempted when available; CPU fallback remains enabled.")
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
        runBusy("Generating speech locally...") {
            val audio = withContext(Dispatchers.IO) { bridge.synthesize(text) }
            setStatus("Generated " + String.format("%.1f", audio.size / 24000f) + " seconds of 24 kHz audio. Playing...")
            playPcm(audio)
            setStatus("Playback completed. QVAC remains loaded for another short request.")
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
        refreshButtons()
        lifecycleScope.launch {
            try {
                action()
            } catch (error: Throwable) {
                setStatus("Operation failed: " + (error.message ?: error.toString()))
            } finally {
                busy = false
                refreshButtons()
            }
        }
    }

    private fun setStatus(message: String) {
        if (::statusView.isInitialized) statusView.text = message
    }

    private fun refreshStatus() {
        setStatus(
            "Files: T3 " + if (t3Ready) "ready" else "missing" +
                " | S3Gen " + if (s3genReady) "ready" else "missing" +
                " | Azure voice " + if (voiceReady) "ready" else "missing" +
                ". Models are not bundled in the APK and are not loaded on app open."
        )
    }

    private fun refreshButtons() {
        if (!::loadButton.isInitialized) return
        loadButton.isEnabled = !busy && t3Ready && s3genReady && voiceReady
        playButton.isEnabled = !busy && engineLoaded
        unloadButton.isEnabled = !busy && engineLoaded
        languagePicker.isEnabled = !busy
        textInput.isEnabled = !busy
    }

    private suspend fun playPcm(samples: FloatArray) = withContext(Dispatchers.IO) {
        require(samples.isNotEmpty()) { "No audio samples to play." }

        val rate = 24000
        val channel = AudioFormat.CHANNEL_OUT_MONO
        val encoding = AudioFormat.ENCODING_PCM_FLOAT
        val minBuffer = AudioTrack.getMinBufferSize(rate, channel, encoding)
        require(minBuffer > 0) { "Android AudioTrack does not support 24 kHz float PCM." }

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(rate)
                    .setChannelMask(channel)
                    .setEncoding(encoding)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBuffer, rate * 4))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        try {
            check(track.state == AudioTrack.STATE_INITIALIZED) { "Android could not initialize AudioTrack." }
            track.play()
            var offset = 0
            while (offset < samples.size) {
                val count = minOf(4096, samples.size - offset)
                val written = track.write(samples, offset, count, AudioTrack.WRITE_BLOCKING)
                check(written > 0) { "AudioTrack write failed: " + written }
                offset += written
            }
            while (track.playbackHeadPosition.toLong() < samples.size.toLong()) delay(20)
            track.stop()
        } finally {
            track.release()
        }
    }
}

private fun copyModelIntoPrivateStorage(context: Context, uri: Uri, targetName: String): File {
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
        input.use { source ->
            FileOutputStream(temporary).use { output ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = source.read(buffer)
                    if (count < 0) break
                    copied += count
                    require(copied <= 8L * 1024L * 1024L * 1024L) { "Model exceeded the 8 GB import limit." }
                    output.write(buffer, 0, count)
                }
                output.flush()
                output.fd.sync()
            }
        }

        require(copied >= 1024L * 1024L) { "Selected file is too small to be a GGUF model." }
        if (target.exists()) check(target.delete()) { "Could not replace previous model file." }
        check(temporary.renameTo(target)) { "Could not finalize copied model file." }
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
                        val magic = ByteArray(6)
                        outputFile.inputStream().use { it.read(magic) }
                        val expectedMagic = byteArrayOf(
                            0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(),
                            'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()
                        )
                        require(magic.contentEquals(expectedMagic)) { name + " is not a NumPy .npy tensor." }
                    }
                    zip.closeEntry()
                    entry = zip.nextEntry
                }
            }
        }

        val missing = requiredProfileFiles - imported
        require(missing.isEmpty()) { "ZIP is missing: " + missing.joinToString() }

        destination.deleteRecursively()
        check(temporary.renameTo(destination)) { "Could not finalize voice profile import." }
    } catch (error: Throwable) {
        temporary.deleteRecursively()
        throw error
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
