package org.apertium.keyboard

import android.app.Application
//import org.apertium.keyboard.di.predictModule
//import org.koin.android.ext.android.startKoin

class SmartKeyboardApp : Application() {

    override fun onCreate() {
        super.onCreate()
        // start Koin!
        //startKoin(this, listOf(predictModule))
    }
}
