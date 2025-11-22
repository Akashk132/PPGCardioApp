package com.example.ppgcardioapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class WaveformView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val paint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 3f
        isAntiAlias = true
    }

    // internal buffer for waveform samples
    private var data: FloatArray = FloatArray(0)

    /**
     * Update the waveform with a new sample array and request redraw.
     * We copy the array to avoid accidental external modification.
     */
    fun updateData(values: FloatArray) {
        data = values.copyOf()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // clear background
        canvas.drawColor(Color.BLACK)

        if (data.isEmpty()) return

        val w = width.toFloat()
        val h = height.toFloat()
        val n = data.size
        val step = if (n > 1) w / (n - 1) else w

        // compute min/max
        var minv = Float.MAX_VALUE
        var maxv = -Float.MAX_VALUE
        for (v in data) {
            if (v < minv) minv = v
            if (v > maxv) maxv = v
        }

        val range = if (maxv - minv < 1e-6f) 1e-6f else (maxv - minv)

        var prevX = 0f
        var prevY = h / 2f

        for (i in data.indices) {
            val x = i * step
            val normalized = (data[i] - minv) / range
            val y = h - (normalized * h)
            canvas.drawLine(prevX, prevY, x, y, paint)
            prevX = x
            prevY = y
        }
    }
}
