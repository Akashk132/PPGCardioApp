package com.example.ppgcardioapp

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import kotlin.math.*
import kotlin.collections.HashMap

class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    // UI
    private lateinit var previewView: PreviewView
    private lateinit var waveformView: WaveformView
    private lateinit var tvHR: TextView
    private lateinit var tvScore: TextView
    private lateinit var tvSQI: TextView
    private lateinit var tvStatus: TextView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var btnFlash: Button
    private lateinit var btnReset: Button

    // Extra parameter TextViews (IBI, HRV, Stress)
    private var tvIBI: TextView? = null
    private var tvHRV: TextView? = null
    private var tvStress: TextView? = null

    // CameraX
    private var cameraControl: CameraControl? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private val executor = Executors.newSingleThreadExecutor()

    // Running flags
    private var running = false
    private var flashOn = false

    // Circular buffer for samples
    private val MAX_BUFFER = 2400 // large buffer (~60-80s at 30-40Hz)
    private val redBuf = FloatArray(MAX_BUFFER)
    private val greenBuf = FloatArray(MAX_BUFFER)
    private val timeBuf = LongArray(MAX_BUFFER)
    private var writePos = 0L
    private var totalSamples = 0

    // Processing window (seconds -> converted to samples using estimated fs)
    private var PROCESS_SECONDS = 8

    // ML model (TFLite)
    private var tflite: Interpreter? = null
    private val TFLITE_NAME = "model.tflite" // put your converted model here (assets)

    // Permission launcher
    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (!granted) runOnUiThread { tvStatus.text = "Camera permission required" }
    }

    // ---------- Reporting / accumulation ----------
    private var reportStartTimeMs: Long = 0L
    private var reporting = false
    private val REPORT_SECONDS = 5L

    // Lists to hold per-window metrics during the 30s reporting window
    private val hrList = ArrayList<Float>()
    private val ibiList = ArrayList<Float>()
    private val hrvList = ArrayList<Float>()
    private val sqiList = ArrayList<Float>()
    private val mlClassList = ArrayList<Int>()
    private val mlScoreList = ArrayList<Float>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // find views that should exist in your layout
        previewView = findViewById(R.id.previewView)
        waveformView = findViewById(R.id.waveformView)
        tvHR = findViewById(R.id.tvHR)
        tvScore = findViewById(R.id.tvScore)
        tvSQI = findViewById(R.id.tvSQI)
        tvStatus = findViewById(R.id.tvStatus)

        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        btnFlash = findViewById(R.id.btnFlash)
        btnReset = findViewById(R.id.btnReset)

        btnStart.setOnClickListener { startCapture() }
        btnStop.setOnClickListener { stopCapture() }
        btnFlash.setOnClickListener { toggleFlash() }
        btnReset.setOnClickListener { resetAll() }

        // attach optional parameter TextViews (IBI, HRV, Stress); create if missing
        safeAttachExtraParameterViews()

        // Permission check
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermission.launch(Manifest.permission.CAMERA)
        } else {
            tvStatus.text = "Ready"
        }

        // Try loading TFLite model if present in assets; if not present, app still works (uses rule-based)
        try {
            tflite = loadTFLiteModelIfExists(this, TFLITE_NAME)
            if (tflite != null) Log.i(TAG, "TFLite model loaded")
            else Log.i(TAG, "TFLite model not found in assets; ML disabled")
        } catch (e: Exception) {
            Log.w(TAG, "Could not load tflite: ${e.localizedMessage}")
            tflite = null
        }
    }

    // If layout lacks extra parameter TextViews, create them and attach below waveform
    private fun safeAttachExtraParameterViews() {
        fun createTv(label: String): TextView {
            val tv = TextView(this)
            tv.layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply {
                (this as LinearLayout.LayoutParams).setMargins(8, 6, 8, 6)
            }
            tv.textSize = 14f
            tv.text = "$label: --"
            return tv
        }

        // Attempt to find pre-existing ids; if missing create and attach
        tvIBI = runCatching { findViewById<TextView>(R.id.tvIBI) }.getOrNull()
        tvHRV = runCatching { findViewById<TextView>(R.id.tvHRV) }.getOrNull()
        tvStress = runCatching { findViewById<TextView>(R.id.tvStress) }.getOrNull()

        // If any missing, create a horizontal layout and append
        if (listOf(tvIBI, tvHRV, tvStress).any { it == null }) {
            val mainLayout = findViewById<LinearLayout>(R.id.main) ?: return
            val container = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                gravity = Gravity.CENTER_HORIZONTAL
            }

            // IBI + HRV + Stress row
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
            }
            if (tvIBI == null) tvIBI = createTv("IBI (s)")
            if (tvHRV == null) tvHRV = createTv("HRV (RMSSD)")
            if (tvStress == null) tvStress = createTv("StressIdx")
            row.addView(tvIBI)
            row.addView(tvHRV)
            row.addView(tvStress)

            container.addView(row)
            mainLayout.addView(container)
        }
    }

    private fun startCapture() {
        if (running) return
        running = true
        btnStart.isEnabled = false
        tvStatus.text = "Starting camera..."
        startCamera()
    }

    private fun stopCapture() {
        if (!running) return
        running = false
        btnStart.isEnabled = true
        tvStatus.text = "Stopped"
        cameraProvider?.unbindAll()
        // reset accumulation
        resetReporting()
    }

    private fun resetAll() {
        writePos = 0
        totalSamples = 0
        for (i in redBuf.indices) redBuf[i] = 0f
        for (i in greenBuf.indices) greenBuf[i] = 0f
        for (i in timeBuf.indices) timeBuf[i] = 0L
        waveformView.updateData(FloatArray(0))
        tvHR.text = "HR: --"
        tvScore.text = "Risk: --"
        tvSQI.text = "SQI: --"
        tvStatus.text = "Status: Reset"
        tvIBI?.text = "IBI (s): --"
        tvHRV?.text = "HRV (RMSSD): --"
        tvStress?.text = "StressIdx: --"
        resetReporting()
    }

    private fun resetReporting() {
        reporting = false
        reportStartTimeMs = 0L
        hrList.clear()
        ibiList.clear()
        hrvList.clear()
        sqiList.clear()
        mlClassList.clear()
        mlScoreList.clear()
    }

    private fun toggleFlash() {
        flashOn = !flashOn
        cameraControl?.enableTorch(flashOn)
        // If available, bump torch strength on Android 13+
        try {
            if (flashOn && Build.VERSION.SDK_INT >= 33) {
                val method = cameraControl?.javaClass?.methods?.firstOrNull { it.name == "setTorchStrengthLevel" }
                method?.invoke(cameraControl, 1)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Torch strength not supported: ${e.localizedMessage}")
        }
        tvStatus.text = if (flashOn) "Flash: ON" else "Flash: OFF"
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build().also { analysis ->
                    analysis.setAnalyzer(executor) { image ->
                        if (!running) { image.close(); return@setAnalyzer }
                        try {
                            val (rAvg, gAvg, bAvg) = averageRGB(image, sampleStride = 3)
                            pushSample(rAvg, gAvg, bAvg)
                        } catch (e: Exception) {
                            Log.e(TAG, "Analyzer error", e)
                        } finally {
                            image.close()
                        }
                    }
                }

            val selector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider?.unbindAll()
                val camera = cameraProvider?.bindToLifecycle(this, selector, preview, imageAnalysis)
                cameraControl = camera?.cameraControl
                tvStatus.text = "Camera running"
            } catch (e: Exception) {
                Log.e(TAG, "Bind failed", e)
                tvStatus.text = "Camera bind failed"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Push sample into circular buffers and trigger processing
    private fun pushSample(r: Float, g: Float, b: Float) {
        val idx = (writePos % MAX_BUFFER).toInt()
        redBuf[idx] = r
        greenBuf[idx] = g
        timeBuf[idx] = System.currentTimeMillis()
        writePos++
        totalSamples = min(totalSamples + 1, MAX_BUFFER)

        val minSamples = 6 * 20 // ~6s @ >=20Hz
        if (totalSamples < minSamples) {
            runOnUiThread {
                tvSQI.text = "SQI: collecting..."
                tvStatus.text = "Place finger & wait"
            }
            return
        }

        val N = determineWindowSamples()
        if (N < 80) return

        // Extract window arrays
        val windowR = FloatArray(N)
        val windowG = FloatArray(N)
        val times = LongArray(N)
        var start = ((writePos - N) % MAX_BUFFER).toInt()
        if (start < 0) start += MAX_BUFFER
        var p = start
        for (i in 0 until N) {
            windowR[i] = redBuf[p]
            windowG[i] = greenBuf[p]
            times[i] = timeBuf[p]
            p++; if (p >= MAX_BUFFER) p = 0
        }

        // estimate sampling rate (Hz)
        val fs = estimateFs(times)

        // Preprocess both signals
        val procR = preprocessWindow(windowR)
        val procG = preprocessWindow(windowG)

        // Hybrid combine: weighted sum. Weighting adapts: more weight to green when red low.
        val weightR = computeRedWeight(windowR, windowG)
        val weightG = 1f - weightR
        val combined = FloatArray(N)
        for (i in 0 until N) combined[i] = weightR * procR[i] + weightG * procG[i]

        // Compute SQI on combined + channel-specific metrics
        val sqiCombined = computeSQI(combined)
        val redGlow = estimateRedGlow(windowR, windowG)
        val pulsatility = computePulsatilityScore(combined, fs)
        val fingerDetected = compositeFingerDetection(windowR, windowG, sqiCombined, pulsatility, redGlow)

        // Update waveform + SQI on UI
        runOnUiThread {
            waveformView.updateData(combined)
            tvSQI.text = "SQI: ${"%.2f".format(sqiCombined)} ${if (fingerDetected) "(finger)" else "(no finger)"}"
        }

        // Start reporting accumulation when fingerDetected true, SQI ok, and flash on
        if (fingerDetected && sqiCombined >= 0.25f && flashOn) {
            if (!reporting) {
                reporting = true
                reportStartTimeMs = System.currentTimeMillis()
                hrList.clear(); ibiList.clear(); hrvList.clear(); sqiList.clear(); mlClassList.clear(); mlScoreList.clear()
                Log.i(TAG, "Reporting started at ${reportStartTimeMs}")
            }
        } else {
            // if finger lost, reset reporting
            if (reporting) {
                // short glitch: don't immediately cancel if it's small — but for simplicity reset
                resetReporting()
                Log.i(TAG, "Reporting reset due to finger lost or low SQI")
            }
        }

        // If finger present & flash on & decent SQI, compute HR/HRV/etc.
        if (fingerDetected && sqiCombined >= 0.28f && flashOn) {
            executor.execute {
                val hrSpec = estimateHRFromSpectrum(combined, fs)
                val peaks = detectPeaks(combined, fs)
                val hrTD = if (peaks.size >= 2) computeHRFromPeaks(peaks, fs) else 0f
                val hr = mergeHrEstimates(hrSpec, hrTD)
                val rr = computeRRFromPeaks(peaks, fs)
                val rmssd = if (rr.isNotEmpty()) computeRMSSD(rr) else 0f
                val ibi = if (rr.isNotEmpty()) rr.average().toFloat() else 0f
                val stressIdx = estimateStressIndex(rr)

                // prepare ML input (resample->detrend->bandpass->normalize) to 512
                val xResampled = resampleTo512(combined, fs)
                val xDet = linearDetrend(xResampled)
                val xBP = bandpassDFT(xDet, 64f, 0.5f, 8.0f)
                val xNorm = zScoreNormalize(xBP)

                val mlResult = predictWithTFLiteIfAvailable(xNorm)
                val score: Float
                val mlClass: Int?
                if (mlResult != null) {
                    mlClass = mlResult.first
                    score = mlResult.second
                } else {
                    mlClass = null
                    score = ruleBasedRisk(hr, rr, sqiCombined)
                }

                // Accumulate values if reporting window active
                if (reporting) {
                    synchronized(this) {
                        if (hr > 0f) hrList.add(hr)
                        if (ibi > 0f) ibiList.add(ibi)
                        if (rmssd > 0f) hrvList.add(rmssd)
                        sqiList.add(sqiCombined)
                        if (mlClass != null) mlClassList.add(mlClass)
                        mlScoreList.add(score)
                    }
                    // check if REPORT_SECONDS elapsed
                    val elapsed = System.currentTimeMillis() - reportStartTimeMs
                    if (elapsed >= REPORT_SECONDS * 1000L) {
                        // copy lists to local to avoid concurrency issues
                        val hrCopy: List<Float>
                        val ibiCopy: List<Float>
                        val hrvCopy: List<Float>
                        val sqiCopy: List<Float>
                        val mlClassCopy: List<Int>
                        val mlScoreCopy: List<Float>
                        synchronized(this) {
                            hrCopy = ArrayList(hrList)
                            ibiCopy = ArrayList(ibiList)
                            hrvCopy = ArrayList(hrvList)
                            sqiCopy = ArrayList(sqiList)
                            mlClassCopy = ArrayList(mlClassList)
                            mlScoreCopy = ArrayList(mlScoreList)
                        }
                        // generate final report (UI)
                        runOnUiThread {
                            try {
                                val report = generateReportText(hrCopy, ibiCopy, hrvCopy, sqiCopy, mlClassCopy, mlScoreCopy)
                                showReportDialog(report, hrCopy, ibiCopy, hrvCopy, sqiCopy, mlClassCopy, mlScoreCopy)
                            } catch (e: Exception) {
                                Log.e(TAG, "Report generation failed: ${e.localizedMessage}")
                            }
                        }
                        // reset reporting after generating report
                        resetReporting()
                    }
                }

                // update UI metrics live
                runOnUiThread {
                    tvHR.text = if (hr > 0f) "HR: ${hr.toInt()}" else "HR: --"
                    tvIBI?.text = "IBI (s): ${if (ibi > 0f) String.format("%.3f", ibi) else "--"}"
                    tvHRV?.text = "HRV (RMSSD): ${if (rmssd > 0f) String.format("%.3f", rmssd) else "--"}"
                    tvStress?.text = "StressIdx: ${String.format("%.2f", stressIdx)}"

                    if (mlClass != null) {
                        val label = when (mlClass) {
                            0 -> "Normal"
                            1 -> "Elevated"
                            2 -> "High Risk"
                            else -> "Unknown"
                        }
                        tvScore.text = "Risk: $label (${String.format("%.2f", score)})"
                        tvStatus.text = when (mlClass) {
                            2 -> "HIGH RISK - seek care"
                            1 -> "MODERATE RISK"
                            0 -> "LOW RISK"
                            else -> "Result ready"
                        }
                    } else {
                        tvScore.text = "Risk: ${"%.2f".format(score)}"
                        tvStatus.text = when {
                            score >= 0.8f -> "HIGH RISK - seek care"
                            score >= 0.5f -> "MODERATE RISK"
                            else -> "LOW RISK"
                        }
                    }
                }
            }
        } else {
            // No reliable finger signal or flash off -> show placeholders
            runOnUiThread {
                tvHR.text = "HR: --"
                tvScore.text = "Risk: --"
                tvIBI?.text = "IBI (s): --"
                tvHRV?.text = "HRV (RMSSD): --"
                tvStress?.text = "StressIdx: --"

                if (!flashOn) tvStatus.text = "Flash must be ON"
                else if (!fingerDetected) tvStatus.text = "No reliable finger signal"
                else tvStatus.text = "Poor SQI - adjust finger/pressure"
            }
        }
    }

    // ===========================
    // === Reporting helpers =====
    // ===========================

    private fun generateReportText(
        hrList: List<Float>,
        ibiList: List<Float>,
        hrvList: List<Float>,
        sqiList: List<Float>,
        mlClassList: List<Int>,
        mlScoreList: List<Float>
    ): String {
        val sb = StringBuilder()
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        sb.append("RuralCardio — 30s PPG Report\n")
        sb.append("Generated: ${sdf.format(Date())}\n\n")
        if (hrList.isEmpty()) {
            sb.append("Insufficient valid PPG windows collected. Try again with proper finger placement and flash ON.\n")
            return sb.toString()
        }

        fun mean(list: List<Float>) = if (list.isEmpty()) 0f else (list.sum() / list.size)
        fun std(list: List<Float>): Float {
            if (list.isEmpty()) return 0f
            val m = mean(list)
            var s = 0.0
            for (v in list) s += (v - m) * (v - m)
            return sqrt((s / list.size).toDouble()).toFloat()
        }

        val hrAvg = mean(hrList)
        val hrStd = std(hrList)
        val ibiAvg = mean(ibiList)
        val ibiStd = std(ibiList)
        val hrvAvg = mean(hrvList)
        val hrvStd = std(hrvList)
        val sqiAvg = mean(sqiList)
        val sqiStd = std(sqiList)
        val mlScoreAvg = if (mlScoreList.isEmpty()) 0f else mlScoreList.sum() / mlScoreList.size

        // majority vote for ML class
        val mlMajority = if (mlClassList.isEmpty()) null else {
            val cnt = HashMap<Int, Int>()
            for (c in mlClassList) cnt[c] = (cnt[c] ?: 0) + 1
            cnt.maxByOrNull { it.value }?.key
        }

        sb.append("Summary (30s window):\n")
        sb.append(" - HR (avg ± sd): ${"%.1f".format(hrAvg)} ± ${"%.1f".format(hrStd)} bpm\n")
        sb.append(" - IBI (avg ± sd): ${"%.3f".format(ibiAvg)} ± ${"%.3f".format(ibiStd)} s\n")
        sb.append(" - HRV (RMSSD avg ± sd): ${"%.3f".format(hrvAvg)} ± ${"%.3f".format(hrvStd)} s\n")
        sb.append(" - SQI (avg ± sd): ${"%.2f".format(sqiAvg)} ± ${"%.2f".format(sqiStd)}\n")
        if (mlMajority != null) {
            val label = when (mlMajority) {
                0 -> "Normal"; 1 -> "Elevated"; 2 -> "High Risk"; else -> "Unknown"
            }
            sb.append(" - ML majority class: $label (avg score ${"%.2f".format(mlScoreAvg)})\n")
        } else {
            sb.append(" - ML  available for majority vote\n")
        }

        sb.append("\nInterpretation & suggestions:\n")

        // interpret HR
        when {
            hrAvg >= 120 -> sb.append("• High heart rate observed (tachycardia range). Recommend immediate clinical review if symptomatic.\n")
            hrAvg >= 100 -> sb.append("• Elevated heart rate (possible stress, fever, or arrhythmia). Re-check and consider medical advice.\n")
            hrAvg >= 60 -> sb.append("• HR in normal resting range.\n")
            else -> sb.append("• Low heart rate observed. If asymptomatic and athletic, may be normal. Otherwise consult clinician.\n")
        }

        // interpret HRV (RMSSD)
        when {
            hrvAvg <= 0.02f -> sb.append("• Very low HRV (possible high sympathetic activity or poor signal). Consider re-test and clinical follow-up.\n")
            hrvAvg <= 0.05f -> sb.append("• Low HRV — may indicate stress or reduced autonomic flexibility.\n")
            else -> sb.append("• HRV within reasonable range.\n")
        }

        // SQI advice
        if (sqiAvg < 0.3f) {
            sb.append("• Signal quality moderate/low (SQI ${"%.2f".format(sqiAvg)}). Reposition finger, ensure flash ON and stable pressure, and retry.\n")
        } else sb.append("• Signal quality adequate.\n")

        if (mlMajority != null && mlMajority == 2) {
            sb.append("• ML flagged HIGH RISK — consider urgent clinical evaluation.\n")
        } else if (mlMajority == 1) {
            sb.append("• ML flagged moderate risk. Consider repeat check and possibly clinical follow-up depending on symptoms.\n")
        } else {
            sb.append("• ML did not flag elevated risk.\n")
        }

        sb.append("\nNotes:\n - This is a screening aid, not a diagnosis.\n - Users with symptoms (chest pain, fainting, severe shortness of breath) should seek immediate care.\n")

        return sb.toString()
    }

    private fun showReportDialog(
        reportText: String,
        hrList: List<Float>,
        ibiList: List<Float>,
        hrvList: List<Float>,
        sqiList: List<Float>,
        mlClassList: List<Int>,
        mlScoreList: List<Float>
    ) {
        val builder = AlertDialog.Builder(this)
            .setTitle("30s PPG Report")
            .setMessage(reportText)
            .setCancelable(true)
            .setPositiveButton("Save PDF") { _, _ ->
                // Save PDF (Downloads preferred)
                try {
                    val fileName = "RuralCardio_Report_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}.pdf"
                    saveReportPdf(reportText, fileName) { success, uriOrMsg ->
                        runOnUiThread {
                            if (success) {
                                val uriStr = uriOrMsg as String
                                AlertDialog.Builder(this)
                                    .setTitle("Saved")
                                    .setMessage("Report saved to: $uriStr")
                                    .setPositiveButton("OK", null)
                                    .show()
                            } else {
                                AlertDialog.Builder(this)
                                    .setTitle("Save failed")
                                    .setMessage(uriOrMsg as String)
                                    .setPositiveButton("OK", null)
                                    .show()
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Save PDF failed: ${e.localizedMessage}")
                }
            }
            .setNegativeButton("Close", null)

        builder.show()
    }

    // Save report as PDF and return via callback(success, uriOrMessage)
    private fun saveReportPdf(reportText: String, fileName: String, callback: (Boolean, Any) -> Unit) {
        // create a PdfDocument and draw text simply across pages
        try {
            val pdf = PdfDocument()
            val pageInfo = PdfDocument.PageInfo.Builder(595, 842, 1).create() // A4-ish
            val paint = Paint()
            paint.textSize = 12f

            // split text into lines and pages
            val lines = reportText.split("\n")
            var currentPage = 1
            var y = 30f
            var page = pdf.startPage(pageInfo)
            var canvas: Canvas = page.canvas
            for (line in lines) {
                if (y > pageInfo.pageHeight - 40) {
                    pdf.finishPage(page)
                    currentPage++
                    page = pdf.startPage(pageInfo)
                    canvas = page.canvas
                    y = 30f
                }
                canvas.drawText(line, 20f, y, paint)
                y += 16f
            }
            pdf.finishPage(page)

            // write PDF to Downloads via MediaStore where possible
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = contentResolver
                val contentValues = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, fileName)
                    put(MediaStore.Downloads.MIME_TYPE, "application/pdf")
                    put(MediaStore.Downloads.IS_PENDING, 1)
                }
                val collection = MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                val uri: Uri? = resolver.insert(collection, contentValues)
                if (uri != null) {
                    resolver.openOutputStream(uri).use { out ->
                        pdf.writeTo(out)
                    }
                    contentValues.clear()
                    contentValues.put(MediaStore.Downloads.IS_PENDING, 0)
                    resolver.update(uri, contentValues, null, null)
                    pdf.close()
                    callback(true, uri.toString())
                    return
                }
            }

            // Fallback: save to internal filesDir
            val outFile = File(filesDir, fileName)
            FileOutputStream(outFile).use { out ->
                pdf.writeTo(out)
            }
            pdf.close()
            callback(true, outFile.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "PDF save error: ${e.localizedMessage}")
            callback(false, "Error saving PDF: ${e.localizedMessage}")
        }
    }

    // ===========================
    // === ML / Preprocess Utils ==
    // ===========================

    // Resample arbitrary-length array 'signal' sampled at inputFs to exactly 512 samples
    private fun resampleTo512(signal: FloatArray, inputFs: Float): FloatArray {
        val targetLen = 512
        val n = signal.size
        if (n == 0) return FloatArray(targetLen) { 0f }
        val out = FloatArray(targetLen)
        val scale = (n - 1).toFloat() / (targetLen - 1)
        for (i in 0 until targetLen) {
            val x = i * scale
            val i0 = floor(x).toInt()
            val i1 = min(i0 + 1, n - 1)
            val t = x - i0
            out[i] = signal[i0] * (1 - t) + signal[i1] * t
        }
        return out
    }

    // Linear detrend (remove best-fit line)
    private fun linearDetrend(x: FloatArray): FloatArray {
        val n = x.size
        if (n < 2) return x.copyOf()
        var sumT = 0.0
        var sumY = 0.0
        var sumTT = 0.0
        var sumTY = 0.0
        for (i in 0 until n) {
            val t = i.toDouble()
            val y = x[i].toDouble()
            sumT += t
            sumY += y
            sumTT += t * t
            sumTY += t * y
        }
        val denom = n * sumTT - sumT * sumT
        val a = if (denom != 0.0) ((n * sumTY - sumT * sumY) / denom) else 0.0
        val b = (sumY - a * sumT) / n
        val out = FloatArray(n)
        for (i in 0 until n) {
            val trend = a * i + b
            out[i] = (x[i] - trend).toFloat()
        }
        return out
    }

    // Bandpass via naive DFT: zero out frequencies outside [low,high] Hz and inverse DFT.
    private fun bandpassDFT(signal: FloatArray, fs: Float, lowHz: Float, highHz: Float): FloatArray {
        val n = signal.size
        if (n == 0) return signal.copyOf()
        val re = DoubleArray(n)
        val im = DoubleArray(n)
        // forward DFT
        for (k in 0 until n) {
            var rsum = 0.0
            var isum = 0.0
            val twoPiKOverN = 2.0 * Math.PI * k / n
            for (t in 0 until n) {
                val ang = twoPiKOverN * t
                val v = signal[t].toDouble()
                rsum += v * cos(ang)
                isum -= v * sin(ang)
            }
            re[k] = rsum
            im[k] = isum
        }
        val outRe = DoubleArray(n) { 0.0 }
        val outIm = DoubleArray(n) { 0.0 }
        for (k in 0 until n) {
            val freq = k.toFloat() * fs / n
            val freqWrap = if (k <= n / 2) freq else (fs - freq)
            if (freqWrap >= lowHz && freqWrap <= highHz) {
                outRe[k] = re[k]
                outIm[k] = im[k]
            }
        }
        // inverse DFT
        val out = FloatArray(n)
        for (t in 0 until n) {
            var rsum = 0.0
            var isum = 0.0
            val twoPiOverN = 2.0 * Math.PI / n
            for (k in 0 until n) {
                val ang = twoPiOverN * k * t
                rsum += outRe[k] * cos(ang) - outIm[k] * sin(ang)
                isum += outRe[k] * sin(ang) + outIm[k] * cos(ang)
            }
            out[t] = (rsum / n).toFloat()
        }
        return out
    }

    // z-score normalization
    private fun zScoreNormalize(x: FloatArray): FloatArray {
        val n = x.size
        if (n == 0) return x.copyOf()
        var mean = 0.0
        for (v in x) mean += v
        mean /= n
        var varsum = 0.0
        for (v in x) {
            val d = v - mean
            varsum += d * d
        }
        val std = sqrt(varsum / n + 1e-12)
        val out = FloatArray(n)
        val denom = if (std < 1e-6) 1e-6 else std
        for (i in 0 until n) out[i] = ((x[i] - mean) / denom).toFloat()
        return out
    }

    // If model.tflite exists in assets, run inference and return pair(classIndex, classProbability)
    private fun predictWithTFLiteIfAvailable(input512: FloatArray): Pair<Int, Float>? {
        val model = tflite ?: return null
        if (input512.size != 512) return null

        // build input in shape [1][1][512]
        val inArr = Array(1) { Array(1) { FloatArray(512) } }
        for (i in 0 until 512) inArr[0][0][i] = input512[i]

        // output array shape [1][3]
        val outArr = Array(1) { FloatArray(3) { 0f } }

        return try {
            model.run(inArr, outArr)
            val probs = outArr[0]
            var best = 0
            var bestP = probs[0]
            for (i in probs.indices) {
                if (probs[i] > bestP) { best = i; bestP = probs[i] }
            }
            Pair(best, bestP)
        } catch (e: Exception) {
            Log.w(TAG, "TFLite run failed: ${e.localizedMessage}")
            null
        }
    }

    // Load TFLite model if exists in assets; otherwise return null.
    private fun loadTFLiteModelIfExists(ctx: Context, assetName: String): Interpreter? {
        return try {
            val list = ctx.assets.list("") ?: arrayOf()
            if (!list.contains(assetName)) return null
            // copy to file
            val outFile = File(ctx.filesDir, assetName)
            ctx.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (input.read(buffer).also { read = it } > 0) output.write(buffer, 0, read)
                    output.flush()
                }
            }
            val mapped = loadMappedFile(outFile)
            val options = Interpreter.Options()
            Interpreter(mapped, options)
        } catch (e: Exception) {
            Log.w(TAG, "Load tflite failed: ${e.localizedMessage}")
            null
        }
    }

    private fun loadMappedFile(file: File): MappedByteBuffer {
        val stream = FileInputStream(file)
        val fc = stream.channel
        val sz = fc.size()
        val mb = fc.map(FileChannel.MapMode.READ_ONLY, 0, sz)
        fc.close()
        stream.close()
        return mb
    }

    // ===========================
    // === Existing helpers kept ==
    // ===========================

    // Fast subsampled YUV->RGB average
    private fun averageRGB(image: ImageProxy, sampleStride: Int = 4): Triple<Float, Float, Float> {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride

        val width = image.width
        val height = image.height

        val yBytes = ByteArray(yBuffer.remaining())
        yBuffer.get(yBytes)
        val uBytes = ByteArray(uBuffer.remaining())
        uBuffer.get(uBytes)
        val vBytes = ByteArray(vBuffer.remaining())
        vBuffer.get(vBytes)

        var rSum = 0L
        var gSum = 0L
        var bSum = 0L
        var count = 0

        val step = max(1, sampleStride)
        for (row in 0 until height step step) {
            for (col in 0 until width step step) {
                val yIndex = row * yRowStride + col
                val y = (yBytes.getOrNull(yIndex)?.toInt() ?: 0) and 0xFF
                val uvRow = row / 2
                val uvCol = col / 2
                val uvIndex = uvRow * uRowStride + uvCol * uPixelStride
                val u = (uBytes.getOrNull(uvIndex)?.toInt() ?: 128) and 0xFF
                val v = (vBytes.getOrNull(uvIndex)?.toInt() ?: 128) and 0xFF

                val yF = y.toFloat()
                val uF = (u - 128).toFloat()
                val vF = (v - 128).toFloat()

                var rr = (yF + 1.402f * vF).roundToInt()
                var gg = (yF - 0.344136f * uF - 0.714136f * vF).roundToInt()
                var bb = (yF + 1.772f * uF).roundToInt()

                rr = rr.coerceIn(0, 255)
                gg = gg.coerceIn(0, 255)
                bb = bb.coerceIn(0, 255)

                rSum += rr
                gSum += gg
                bSum += bb
                count++
            }
        }

        if (count == 0) return Triple(0f, 0f, 0f)
        return Triple(rSum.toFloat() / count, gSum.toFloat() / count, bSum.toFloat() / count)
    }

    // Determine window sample count from PROCESS_SECONDS and estimated fs (fallback to 30)
    private fun determineWindowSamples(): Int {
        val recent = min(totalSamples, 200)
        if (recent < 10) return PROCESS_SECONDS * 25
        var cnt = 0
        var sumDt = 0L
        var p = ((writePos - 1) % MAX_BUFFER).toInt()
        if (p < 0) p += MAX_BUFFER
        var prev = timeBuf[p]
        for (i in 1..recent) {
            p--; if (p < 0) p += MAX_BUFFER
            val t = timeBuf[p]
            val dt = prev - t
            if (dt > 0) {
                sumDt += dt
                cnt++
            }
            prev = t
        }
        val avgDt = if (cnt > 0) sumDt / cnt.toDouble() else 33.3
        val fs = if (avgDt > 0) (1000.0 / avgDt).toFloat() else 30f
        val samples = (PROCESS_SECONDS * fs).toInt()
        return samples.coerceIn(80, min(totalSamples, MAX_BUFFER))
    }

    // Estimate sampling freq from time array
    private fun estimateFs(times: LongArray): Float {
        if (times.size < 2) return 30f
        var cnt = 0
        var sum = 0L
        for (i in 1 until times.size) {
            val dt = times[i] - times[i - 1]
            if (dt > 0) {
                sum += dt
                cnt++
            }
        }
        if (cnt == 0) return 30f
        val avgMs = sum.toDouble() / cnt.toDouble()
        return (1000.0 / avgMs).toFloat()
    }

    // Detrend + smooth + normalize (kept for local visualization)
    private fun preprocessWindow(win: FloatArray): FloatArray {
        val n = win.size
        val out = FloatArray(n)
        if (n == 0) return out
        val maLen = max(4, n / 8)
        val ma = FloatArray(n)
        var s = 0f
        for (i in 0 until n) {
            s += win[i]
            if (i >= maLen) {
                s -= win[i - maLen]
                ma[i] = s / maLen
            } else ma[i] = s / (i + 1)
        }
        for (i in 0 until n) out[i] = win[i] - ma[i]
        val smooth = FloatArray(n)
        val k = max(1, n / 60)
        for (i in 0 until n) {
            var sum = 0f
            var cnt = 0
            for (j in max(0, i - k)..min(n - 1, i + k)) {
                sum += out[j]; cnt++
            }
            smooth[i] = sum / cnt
        }
        var mean = 0f
        for (v in smooth) mean += v
        mean /= n
        var varsum = 0f
        for (v in smooth) varsum += (v - mean) * (v - mean)
        val std = sqrt(varsum / n + 1e-9f)
        for (i in 0 until n) smooth[i] = (smooth[i] - mean) / (std + 1e-6f)
        return smooth
    }

    // Compute spectral pulsatility ratio
    private fun computePulsatilityScore(proc: FloatArray, fs: Float): Float {
        val n = proc.size
        if (n < 16) return 0f
        val maxK = n / 2
        var totalPower = 0.0
        var peakPower = 0.0
        for (k in 1..maxK) {
            var re = 0.0
            var im = 0.0
            val freq = k.toFloat() * fs / n
            for (t in 0 until n) {
                val ang = -2.0 * Math.PI * k * t / n
                val v = proc[t].toDouble()
                re += v * cos(ang)
                im += v * sin(ang)
            }
            val p = re * re + im * im
            totalPower += p
            if (freq in 0.7f..4.0f) {
                if (p > peakPower) peakPower = p
            }
        }
        if (totalPower <= 0.0) return 0f
        val ratio = (peakPower / totalPower).toFloat()
        return ratio.coerceIn(0f, 1f)
    }

    // SQI: combination of variance and autocorrelation proxy
    private fun computeSQI(proc: FloatArray): Float {
        if (proc.isEmpty()) return 0f
        val n = proc.size
        var varsum = 0f
        for (v in proc) varsum += v * v
        val variance = varsum / n
        val lag = max(1, n / 8)
        var ac = 0f
        var cnt = 0
        for (i in 0 until n - lag) {
            ac += proc[i] * proc[i + lag]
            cnt++
        }
        ac = if (cnt > 0) ac / cnt else 0f
        val vscore = (tanh(variance.toDouble()).toFloat()).coerceIn(0f, 1f)
        val pscore = ((ac + 1f) / 2f).coerceIn(0f, 1f)
        return (0.55f * vscore + 0.45f * pscore).coerceIn(0f, 1f)
    }

    // Detect red glow
    private fun estimateRedGlow(winR: FloatArray, winG: FloatArray): Float {
        val n = winR.size
        if (n == 0) return 0f
        val tail = max(1, n / 6)
        var rsum = 0f
        var gsum = 0f
        for (i in n - tail until n) {
            rsum += winR[i]
            gsum += winG[i]
        }
        val rMean = rsum / tail
        val gMean = gsum / tail
        val ratio = rMean / (gMean + 1e-6f)
        return ratio
    }

    private fun computeRedWeight(winR: FloatArray, winG: FloatArray): Float {
        val ratio = estimateRedGlow(winR, winG)
        val w = ((ratio - 1.0f) * 0.6f).coerceIn(0f, 0.6f)
        return w
    }

    private fun compositeFingerDetection(
        winR: FloatArray,
        winG: FloatArray,
        sqi: Float,
        pulsatility: Float,
        redGlow: Float
    ): Boolean {

        val n = winR.size
        if (n < 20) return false

        // 1) RED CHANNEL MUST BE DOMINANT
        val meanR = winR.average().toFloat()
        val meanG = winG.average().toFloat()
        val redRatio = meanR / (meanG + 1e-6f)

        val redOk = redRatio > 2.0f   // require strong red dominance

        // 2) RED VARIANCE SHOULD NOT BE ZERO (finger gives small pulse variation)
        val varR = winR.map { it - meanR }.map { it * it }.average().toFloat()
        val varOk = varR > 0.3f && varR < 80f

        // 3) SQI must be somewhat OK
        val sqiOk = sqi > 0.15f

        // 4) Pulsatility only needed AFTER steady signal
        val pulseOk = pulsatility > 0.05f

        // 5) Flash must be ON
        val flashOk = flashOn

        return redOk && varOk && sqiOk && pulseOk && flashOk
    }

    // Peak detection (time-domain)
    private fun detectPeaks(signal: FloatArray, fs: Float): List<Int> {
        val peaks = ArrayList<Int>()
        if (signal.size < 3) return peaks
        val n = signal.size
        var mean = 0f
        for (v in signal) mean += v
        mean /= n
        var varsum = 0f
        for (v in signal) varsum += (v - mean) * (v - mean)
        val std = sqrt(varsum / n + 1e-9f)
        val threshold = mean + max(0.2f, 0.6f * std)
        val minInterval = max(1, (0.35f * fs).toInt())
        var lastPeak = -minInterval
        for (i in 1 until n - 1) {
            val v = signal[i]
            if (v > signal[i - 1] && v >= signal[i + 1] && v > threshold) {
                if (i - lastPeak >= minInterval) {
                    peaks.add(i)
                    lastPeak = i
                }
            }
        }
        return peaks
    }

    private fun computeHRFromPeaks(peaks: List<Int>, fs: Float): Float {
        if (peaks.size < 2) return 0f
        val durations = FloatArray(peaks.size - 1)
        for (i in 0 until peaks.size - 1) durations[i] = (peaks[i + 1] - peaks[i]) / fs
        val meanRR = durations.average().toFloat()
        if (meanRR <= 0f) return 0f
        return (60f / meanRR)
    }

    private fun computeRRFromPeaks(peaks: List<Int>, fs: Float): FloatArray {
        if (peaks.size < 2) return FloatArray(0)
        val rr = FloatArray(peaks.size - 1)
        for (i in 0 until peaks.size - 1) rr[i] = (peaks[i + 1] - peaks[i]) / fs
        return rr
    }

    private fun computeRMSSD(rr: FloatArray): Float {
        if (rr.size < 2) return 0f
        var s = 0f
        for (i in 0 until rr.size - 1) {
            val d = rr[i + 1] - rr[i]
            s += d * d
        }
        return sqrt(s / (rr.size - 1))
    }

    // Spectral HR estimation
    private fun estimateHRFromSpectrum(proc: FloatArray, fs: Float): Float {
        val n = proc.size
        if (n < 16) return 0f
        var bestFreq = 0f
        var bestP = 0.0
        val maxK = n / 2
        for (k in 1..maxK) {
            var re = 0.0
            var im = 0.0
            val freq = k.toFloat() * fs / n
            if (freq < 0.5f || freq > 4.5f) continue
            for (t in 0 until n) {
                val ang = -2.0 * Math.PI * k * t / n
                val v = proc[t].toDouble()
                re += v * cos(ang)
                im += v * sin(ang)
            }
            val p = re * re + im * im
            if (p > bestP) {
                bestP = p
                bestFreq = freq
            }
        }
        return if (bestFreq > 0f) bestFreq * 60f else 0f
    }

    private fun mergeHrEstimates(hrSpec: Float, hrTD: Float): Float {
        if (hrSpec <= 0f && hrTD <= 0f) return 0f
        if (hrSpec <= 0f) return hrTD
        if (hrTD <= 0f) return hrSpec
        return if (abs(hrSpec - hrTD) <= 8f) (hrSpec + hrTD) / 2f else (0.7f * hrSpec + 0.3f * hrTD)
    }

    // Small rule based stress index from RR intervals
    private fun estimateStressIndex(rr: FloatArray): Float {
        if (rr.isEmpty()) return 0f
        val rmssd = computeRMSSD(rr)
        val idx = 1f - (rmssd / (rmssd + 0.05f)) // scale to 0..1
        return idx.coerceIn(0f, 1f)
    }

    // Simple conservative rule-based risk (kept)
    private fun ruleBasedRisk(hr: Float, rr: FloatArray, sqi: Float): Float {
        if (hr <= 0f || rr.isEmpty()) return 0f
        val hrScore = when {
            hr >= 140 -> 1f
            hr >= 120 -> 0.8f
            hr >= 100 -> 0.4f
            else -> 0f
        }
        val rmssd = if (rr.size >= 2) computeRMSSD(rr) else 0f
        val hrvScore = (1f - (rmssd / 0.2f)).coerceIn(0f, 1f)
        val noisePenalty = (1f - sqi).coerceIn(0f, 1f)
        return (0.6f * hrScore + 0.3f * hrvScore + 0.1f * noisePenalty).coerceIn(0f, 1f)
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        tflite?.close()
    }
}
