/**
 * secure_storage.js — Encrypted Key-Value Wrapper
 * Secures JWT tokens, API keys using Expo SecureStore (iOS Keychain / Android Keystore).
 */

import * as SecureStore from 'expo-secure-store';

/**
 * Saves a sensitive value securely.
 * 
 * @param {string} key 
 * @param {string} value 
 * @returns {Promise<boolean>}
 */
export async function saveSecureItem(key, value) {
    try {
        if (!value) return false;
        
        // Keychain allows ~4kb limits typically. JWTs easily fit.
        // We enforce HIGHEST security level available.
        await SecureStore.setItemAsync(key, value, {
            keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
        });
        console.log(`[VAULT] Saved key securely: ${key}`);
        return true;
    } catch (error) {
        console.error(`[VAULT] Failed to save key ${key}:`, error);
        return false;
    }
}

/**
 * Retrieves a securely stored value.
 * 
 * @param {string} key 
 * @returns {Promise<string|null>} The stored string or null.
 */
export async function getSecureItem(key) {
    try {
        const result = await SecureStore.getItemAsync(key);
        return result || null;
    } catch (error) {
        console.error(`[VAULT] Failed to retrieve key ${key}:`, error);
        return null;
    }
}

/**
 * Deletes a secure key (useful for logout).
 * 
 * @param {string} key 
 * @returns {Promise<boolean>}
 */
export async function deleteSecureItem(key) {
    try {
        await SecureStore.deleteItemAsync(key);
        console.log(`[VAULT] Deleted secure key: ${key}`);
        return true;
    } catch (error) {
        console.error(`[VAULT] Failed to delete key ${key}:`, error);
        return false;
    }
}

/**
 * Clears entirely the auth bundle (access, refresh, and session ID).
 * Called immediately upon `Auth.execute_logout()` success.
 */
export async function wipeSessionData() {
    await deleteSecureItem('apexalgo_access_token');
    await deleteSecureItem('apexalgo_refresh_token');
    await deleteSecureItem('apexalgo_session_id');
    console.log('[VAULT] Complete session wipe executed.');
}
