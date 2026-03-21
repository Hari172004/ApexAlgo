/**
 * biometric.js — Fingerprint and Face ID Authentication 
 * Uses expo-local-authentication to quickly secure the ApexAlgo mobile app.
 */

import * as LocalAuthentication from 'expo-local-authentication';

/**
 * Prompts the user for biometric authentication.
 * Used when app comes from background, or at initial login screen.
 * 
 * @param {string} promptMessage The text displayed on the Face ID / Touch ID popup.
 * @returns {Promise<boolean>} True if authenticated, false otherwise.
 */
export async function authenticateBiometric(promptMessage = 'Unlock ApexAlgo Trading') {
    try {
        // 1. Check if hardware exists
        const hasHardware = await LocalAuthentication.hasHardwareAsync();
        if (!hasHardware) {
            console.warn('[BIOMETRIC] Device does not have biometric hardware.');
            return false;
        }

        // 2. Check if the user has fingerprints/face enrolled
        const isEnrolled = await LocalAuthentication.isEnrolledAsync();
        if (!isEnrolled) {
            console.warn('[BIOMETRIC] No biometrics enrolled on this device.');
            return false;
        }

        // 3. Prompt user
        const result = await LocalAuthentication.authenticateAsync({
            promptMessage,
            fallbackLabel: 'Use Passcode',
            disableDeviceFallback: false,
            cancelLabel: 'Cancel',
        });

        if (result.success) {
            console.log('[BIOMETRIC] Authentication successful.');
            return true;
        } else {
            console.warn(`[BIOMETRIC] Authentication failed: ${result.error}`);
            // e.g. user_canceled, authentication_failed
            return false;
        }
    } catch (error) {
        console.error('[BIOMETRIC] Critical error during biometric prompt:', error);
        return false;
    }
}

/**
 * Returns an array of supported biometric types (e.g. FACIAL_RECOGNITION, FINGERPRINT, IRIS)
 */
export async function getSupportedBiometrics() {
    try {
        const types = await LocalAuthentication.supportedAuthenticationTypesAsync();
        return types;
    } catch (error) {
        console.error('[BIOMETRIC] Failed to get supported types:', error);
        return [];
    }
}
