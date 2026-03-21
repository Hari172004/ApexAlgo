/**
 * root_detection.js — Jailbreak & Root Check
 * Blocks the ApexAlgo app on compromised devices to prevent 
 * broker credential dumping or API reverse engineering.
 */

import * as Device from 'expo-device';
import { Alert, BackHandler, Platform } from 'react-native';

/**
 * Checks if the Android or iOS device is rooted / jailbroken.
 * Expo device info provides an experimental async method.
 * 
 * @returns {Promise<boolean>} True if device is compromised.
 */
export async function isDeviceCompromised() {
    try {
        // Only supported on Android/iOS natively
        if (Platform.OS === 'web') return false;

        const isRooted = await Device.isRootedExperimentalAsync();
        
        if (isRooted) {
            console.error('[SECURITY_SHIELD] Root/Jailbreak detected on device!');
            return true;
        }

        return false;
    } catch (error) {
        console.error('[SECURITY_SHIELD] Failed to check root status:', error);
        // Fail open or closed? Typically fail open in consumer apps if check crashes.
        return false;
    }
}

/**
 * Enforces the check on app startup. 
 * If compromised, displays a warning and forces the app to close.
 */
export async function enforceDeviceIntegrity() {
    const compromised = await isDeviceCompromised();
    
    if (compromised) {
        Alert.alert(
            "Security Violation",
            "This device appears to be rooted or jailbroken. For your financial security and to protect broker API keys, ApexAlgo Trading is disabled on this device.",
            [
                { 
                    text: "Exit Application", 
                    onPress: () => {
                        if (Platform.OS === 'android') {
                            BackHandler.exitApp();
                        } else {
                            // On iOS, force exiting app is frowned upon by Apple, 
                            // but for critical financial apps, freezing navigation 
                            // or intentionally throwing works.
                            throw new Error("Compromised Device Halted");
                        }
                    } 
                }
            ],
            { cancelable: false }
        );
        return false;
    }
    
    console.log('[SECURITY_SHIELD] Device integrity verified successfully.');
    return true;
}
