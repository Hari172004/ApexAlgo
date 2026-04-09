import React, { useState, useEffect, createContext, useContext } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, ScrollView,
  TextInput, Alert, ActivityIndicator, StatusBar, Platform
} from 'react-native';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Activity, Settings, History, TrendingUp, ShieldAlert, ArrowUpRight, ArrowDownRight, LogOut, Play, Square } from 'lucide-react-native';

// ── Theme (Premium Dark / Gold) ───────────────────────────────
const T = {
  bg:      '#08080C',
  surface: '#12121A',
  card:    '#1A1A24',
  border:  '#2C2C3E',
  gold:    '#D4AF37',
  goldDim: '#D4AF3722',
  text:    '#F2F2F5',
  muted:   '#8A8A9E',
  green:   '#00E676',
  red:     '#FF3B30',
};

const NavTheme = {
  ...DefaultTheme,
  colors: { ...DefaultTheme.colors, background: T.bg },
};

// ── API Base ──────────────────────────────────────────────────
// Automatically uses the environment variable if you start Expo with EXPO_PUBLIC_API_URL=...
// Otherwise defaults to the local PC connection for simulator dev.
const API_BASE = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8086';

// ── Context ───────────────────────────────────────────────────
const AuthCtx = createContext(null);
function useAuth() { return useContext(AuthCtx); }

// ── Shared API ────────────────────────────────────────────────
async function apiCall(endpoint, method = 'GET', body = null, token = '') {
  try {
    const opts = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    };
    
    // For local dev with Expo emulator, use localhost. (On physical device, replace with PC local IP)
    const url = `${API_BASE}${endpoint}`;
    
    const response = await fetch(url, opts);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || data.message || data.error || 'API Request Failed');
    }
    
    return data;
  } catch(e) {
    console.log(`API Error (${endpoint}):`, e.message);
    throw new Error(`${e.message}`);
  }
}

// ── LOGIN SCREEN ──────────────────────────────────────────────
function LoginScreen({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) return Alert.alert('Error', 'Fill in all fields');
    setLoading(true);
    try {
      // Simulate real server delay
      setTimeout(() => {
        if (email.toLowerCase() !== 'admin@agniv.com' || password !== 'Admin123') {
          Alert.alert('Authentication Failed', 'Invalid email or password provided.');
          setLoading(false);
          return;
        }
        onLogin({ uid: 'admin_101', email, token: 'mockAdminToken' });
        setLoading(false);
      }, 1000);
    } catch (e) {
      Alert.alert('Login Failed', e.message);
      setLoading(false);
    }
  };

  return (
    <LinearGradient colors={['#1A1A24', '#08080C']} style={s.screen}>
      <View style={s.loginBox}>
        <View style={{ alignItems: 'center', marginBottom: 40 }}>
          <View style={s.iconWrapper}>
            <Activity color={T.gold} size={48} strokeWidth={2.5} />
          </View>
          <Text style={s.logo}>AGNIV</Text>
          <Text style={s.subtitle}>Trade Above the Market</Text>
        </View>

        <View style={s.inputContainer}>
          <TextInput style={s.input} placeholder="Email" placeholderTextColor={T.muted}
            value={email} onChangeText={setEmail} keyboardType="email-address" autoCapitalize="none" />
        </View>

        <View style={s.inputContainer}>
          <TextInput style={s.input} placeholder="Password" placeholderTextColor={T.muted}
            value={password} onChangeText={setPassword} secureTextEntry />
        </View>

        <TouchableOpacity style={s.btnGold} onPress={handleLogin} disabled={loading} activeOpacity={0.8}>
          {loading ? <ActivityIndicator color={T.bg} /> : <Text style={s.btnGoldText}>SIGN IN</Text>}
        </TouchableOpacity>
        
        <Text style={[s.muted, { textAlign: 'center', marginTop: 30 }]}>
          Don't have an account? <Text style={{color: T.gold}}>Sign Up via Web</Text>
        </Text>
      </View>
    </LinearGradient>
  );
}

// ── CUSTOM COMPONENTS ─────────────────────────────────────────
const Card = ({ children, style }) => (
  <View style={[s.card, style]}>{children}</View>
);

const StatMini = ({ label, value, color = T.text }) => (
  <View style={{ flex: 1, alignItems: 'center' }}>
    <Text style={s.statLabel}>{label}</Text>
    <Text style={[s.statVal, { color }]}>{value}</Text>
  </View>
);

// ── DASHBOARD SCREEN ──────────────────────────────────────────
function DashboardScreen() {
  const { user } = useAuth();
  const insets = useSafeAreaInsets();
  const [status, setStatus] = useState(null);
  const [reloading, setReloading] = useState(false);

  const fetchStatus = async () => {
    try { setStatus((await apiCall('/bot/status', 'GET', null, user.token))); }
    catch (e) { console.log(e.message); }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const toggleBot = async (start) => {
    setReloading(true);
    try {
      if (start) {
        await apiCall('/bot/start', 'POST', { user_id: user.uid, config: { mode: 'DEMO', assets: 'BOTH', strategy: 'AUTO' } }, user.token);
        Alert.alert('✅ Active', 'Bot engine started!');
      } else {
        await apiCall('/bot/stop', 'POST', null, user.token);
        Alert.alert('🛑 Stopped', 'Bot engine halted.');
      }
      await fetchStatus();
    } catch (e) { Alert.alert('Error', e.message); }
    finally { setReloading(false); }
  };

  const running = status?.running || false;
  const balance = status?.balance || 10000;

  return (
    <ScrollView style={s.screen} contentContainerStyle={[s.scrollContent, { paddingTop: insets.top + 20 }]}>
      <View style={s.headerRow}>
        <Text style={s.pageTitle}>Dashboard</Text>
        <View style={s.statusBadge(running)}>
          <View style={[s.statusDot, { backgroundColor: running ? T.green : T.red }]} />
          <Text style={s.statusBadgeText}>{running ? 'ONLINE' : 'OFFLINE'}</Text>
        </View>
      </View>

      <LinearGradient colors={['#D4AF3715', '#08080C']} style={s.balanceCard}>
        <Text style={s.balanceLabel}>Total Equity ({status?.mode || 'DEMO'})</Text>
        <Text style={s.balanceValue}>
          <Text style={{fontSize: 24, color: T.gold}}>$$</Text>
          {balance.toLocaleString('en', { minimumFractionDigits: 2 })}
        </Text>
      </LinearGradient>

      <View style={s.statsRow}>
        <Card style={{ flex: 1, marginRight: 12, paddingVertical: 16 }}>
          <StatMini label="Win Rate" value={`${status?.risk_stats?.win_rate_today ?? 0}%`} color={T.green} />
        </Card>
        <Card style={{ flex: 1, paddingVertical: 16 }}>
          <StatMini label="Trades (Today)" value={status?.risk_stats?.trade_count_today ?? 0} />
        </Card>
      </View>

      {status?.funded_report && (
        <Card>
          <View style={s.rowBetween}>
            <View style={s.row}>
              <ShieldAlert color={T.gold} size={16} />
              <Text style={[s.cardTitle, { marginLeft: 8 }]}>{status.funded_report.firm}</Text>
            </View>
            <Text style={s.muted}>{status.funded_report.phase}</Text>
          </View>
          <View style={{ marginTop: 12 }}>
            <View style={s.rowBetween}>
              <Text style={s.mutedSmall}>Profit Target</Text>
              <Text style={{color: T.green, fontSize: 13, fontWeight:'600'}}>{status.funded_report.profit_progress_pct?.toFixed(1)}%</Text>
            </View>
            <View style={s.progressBar}><View style={[s.progressFill, { width: `${Math.min(100, status.funded_report.profit_progress_pct || 0)}%` }]} /></View>
          </View>
          <View style={{ marginTop: 12 }}>
            <View style={s.rowBetween}>
              <Text style={s.mutedSmall}>Drawdown Used</Text>
              <Text style={{color: T.red, fontSize: 13, fontWeight:'600'}}>{status.funded_report.drawdown_used_pct?.toFixed(1)}%</Text>
            </View>
            <View style={s.progressBar}><View style={[s.progressFill, { backgroundColor: T.red, width: `${Math.min(100, status.funded_report.drawdown_used_pct || 0)}%` }]} /></View>
          </View>
        </Card>
      )}

      <View style={[s.rowBetween, { marginTop: 10 }]}>
        <TouchableOpacity style={[s.actionBtn, s.startBtn, running && s.btnDisabled]} onPress={() => toggleBot(true)} disabled={running || reloading} activeOpacity={0.7}>
          <Play color={T.bg} size={20} fill={T.bg} style={{marginRight: 8}} />
          <Text style={s.actionBtnText}>START BOT</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[s.actionBtn, s.stopBtn, !running && s.btnDisabled]} onPress={() => toggleBot(false)} disabled={!running || reloading} activeOpacity={0.7}>
          <Square color={T.text} size={18} fill={T.text} style={{marginRight: 8}} />
          <Text style={[s.actionBtnText, {color: T.text}]}>STOP BOT</Text>
        </TouchableOpacity>
      </View>

      {status?.open_positions?.length > 0 && (
        <View style={{ marginTop: 24 }}>
          <Text style={s.sectionTitle}>Live Positions ({status.open_positions.length})</Text>
          {status.open_positions.map((pos, i) => {
            const isBuy = pos.direction === 'BUY';
            const prof = pos.pnl >= 0;
            return (
              <Card key={i} style={s.posCard}>
                <View style={s.rowBetween}>
                  <View style={s.row}>
                    <View style={[s.posIcon, { backgroundColor: isBuy ? '#00E67622' : '#FF3B3022' }]}>
                      {isBuy ? <ArrowUpRight color={T.green} size={18} /> : <ArrowDownRight color={T.red} size={18} />}
                    </View>
                    <View style={{ marginLeft: 12 }}>
                      <Text style={s.posSymbol}>{pos.symbol}</Text>
                      <Text style={[s.posVol, { color: isBuy ? T.green : T.red }]}>{pos.direction} • {pos.volume} Lots</Text>
                    </View>
                  </View>
                  <View style={{ alignItems: 'flex-end' }}>
                    <Text style={[s.posPnl, { color: prof ? T.green : T.red }]}>{prof ? '+' : '-'}${Math.abs(pos.pnl || 0).toFixed(2)}</Text>
                    <Text style={s.posPrice}>SL: {pos.sl?.toFixed(2)}</Text>
                  </View>
                </View>
              </Card>
            )})}
        </View>
      )}
      <View style={{height: 40}} />
    </ScrollView>
  );
}

// ── SETTINGS SCREEN ───────────────────────────────────────────
function SettingsScreen() {
  const { setUser } = useAuth();
  const insets = useSafeAreaInsets();
  
  const [mode, setMode] = useState('DEMO');
  const [strategy, setStrategy] = useState('AUTO');
  const [assets, setAssets] = useState('BOTH');
  
  const [mt5Id, setMt5Id] = useState('');
  const [mt5Pass, setMt5Pass] = useState('');
  const [mt5Server, setMt5Server] = useState('');
  const [savingBroker, setSavingBroker] = useState(false);

  const saveBrokerCredentials = async () => {
    if (!mt5Id || !mt5Pass || !mt5Server) {
        Alert.alert('Error', 'Please fill out Account ID, Password, and Server.');
        return;
    }
    setSavingBroker(true);
    try {
      await apiCall('/bot/config', 'POST', { 
          mt5_account: mt5Id, 
          mt5_password: mt5Pass, 
          mt5_server: mt5Server 
      });
      Alert.alert('✅ Credentials Saved', 'Your MT5 account is securely linked to the cloud engine.');
    } catch (e) {
      Alert.alert('Error connecting to backend', e.message);
    } finally {
      setSavingBroker(false);
    }
  };

  const ToggleGroup = ({ label, options, value, onChange }) => (
    <View style={{ marginBottom: 24 }}>
      <Text style={[s.cardTitle, { marginBottom: 12 }]}>{label}</Text>
      <View style={{flexDirection: 'row', flexWrap: 'wrap', gap: 10}}>
        {options.map(opt => (
          <TouchableOpacity key={opt} activeOpacity={0.7}
            style={[s.chip, value === opt && s.chipActive]}
            onPress={() => onChange(opt)}>
            <Text style={[s.chipText, value === opt && s.chipTextActive]}>{opt}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  return (
    <View style={[s.screen, { paddingTop: insets.top + 20 }]}>
      <View style={[s.headerRow, { paddingHorizontal: 20 }]}>
        <Text style={s.pageTitle}>Configuration</Text>
      </View>
      <ScrollView contentContainerStyle={s.scrollContent}>
        <Card>
          <ToggleGroup label="Trading Mode" options={['DEMO', 'REAL', 'FUNDED']} value={mode} onChange={setMode} />
          <ToggleGroup label="Strategy Engine" options={['AUTO', 'SCALP', 'SWING']} value={strategy} onChange={setStrategy} />
          <ToggleGroup label="Traded Assets" options={['BOTH', 'XAUUSD']} value={assets} onChange={setAssets} />
        </Card>

        <Card style={{marginTop: 16}}>
          <Text style={[s.cardTitle, {marginBottom: 16}]}>Risk Management</Text>
          <View style={s.inputContainer}>
            <TextInput style={s.input} value="1.5" keyboardType="decimal-pad" />
            <Text style={s.inputSuffix}>% Risk</Text>
          </View>
        </Card>

        <Card style={{marginTop: 16, marginBottom: 16}}>
          <Text style={[s.cardTitle, {marginBottom: 16}]}>Broker Connection (MT5)</Text>
          <View style={s.inputContainer}>
            <TextInput style={[s.input, {padding: 14}]} placeholder="Account ID (e.g. 334986967)" placeholderTextColor={T.muted} value={mt5Id} onChangeText={setMt5Id} keyboardType="numeric" />
          </View>
          <View style={s.inputContainer}>
            <TextInput style={[s.input, {padding: 14}]} placeholder="Password" placeholderTextColor={T.muted} value={mt5Pass} onChangeText={setMt5Pass} secureTextEntry />
          </View>
          <View style={[s.inputContainer, {marginBottom: 16}]}>
            <TextInput style={[s.input, {padding: 14}]} placeholder="Server (e.g. XMGlobal-MT5 3)" placeholderTextColor={T.muted} value={mt5Server} onChangeText={setMt5Server} autoCapitalize="none" />
          </View>
          <TouchableOpacity style={[s.btnGold, {marginTop: 0, padding: 14, borderRadius: 12}]} onPress={saveBrokerCredentials} disabled={savingBroker}>
            {savingBroker ? <ActivityIndicator color={T.bg} size="small" /> : <Text style={[s.btnGoldText, {fontSize: 14}]}>Connect Broker</Text>}
          </TouchableOpacity>
        </Card>

        <TouchableOpacity style={s.btnLogout} onPress={() => setUser(null)}>
          <LogOut color={T.red} size={18} />
          <Text style={s.btnLogoutText}>Sign Out / Disconnect</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

// ── HISTORY SCREEN ────────────────────────────────────────────
function HistoryScreen() {
  const insets = useSafeAreaInsets();
  
  return (
    <View style={[s.screen, { paddingTop: insets.top + 20, paddingHorizontal: 20 }]}>
      <Text style={s.pageTitle}>Trade History</Text>
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <History color={T.border} size={64} style={{marginBottom: 16}} />
        <Text style={s.muted}>No trades recorded yet.</Text>
        <Text style={[s.mutedSmall, {marginTop: 8, textAlign: 'center'}]}>Trades will appear here once the engine executes market orders.</Text>
      </View>
    </View>
  );
}

// ── TAB NAVIGATOR ─────────────────────────────────────────────
const Tab = createBottomTabNavigator();

function MainTabs() {
  const insets = useSafeAreaInsets();
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: { 
          backgroundColor: '#12121AD9', borderTopWidth: 0, 
          height: 65 + insets.bottom, paddingTop: 10, paddingBottom: Math.max(insets.bottom, 10),
          position: 'absolute', elevation: 0
        },
        tabBarActiveTintColor: T.gold,
        tabBarInactiveTintColor: T.muted,
        tabBarLabelStyle: { fontSize: 11, fontWeight: '600', marginTop: 4 },
      }}>
      <Tab.Screen name="Dashboard" component={DashboardScreen} options={{ tabBarIcon: ({color}) => <TrendingUp color={color} size={22} /> }} />
      <Tab.Screen name="History"   component={HistoryScreen}   options={{ tabBarIcon: ({color}) => <History color={color} size={22} /> }} />
      <Tab.Screen name="Settings"  component={SettingsScreen}  options={{ tabBarIcon: ({color}) => <Settings color={color} size={22} /> }} />
    </Tab.Navigator>
  );
}

// ── ROOT APP ──────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState(null);

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
      <AuthCtx.Provider value={{ user, setUser }}>
        <NavigationContainer theme={NavTheme}>
          {!user ? <LoginScreen onLogin={setUser} /> : <MainTabs />}
        </NavigationContainer>
      </AuthCtx.Provider>
    </SafeAreaProvider>
  );
}

// ── STYLES ────────────────────────────────────────────────────
const s = StyleSheet.create({
  screen: { flex: 1, backgroundColor: T.bg },
  scrollContent: { padding: 20 },
  
  // Login
  loginBox: { flex: 1, justifyContent: 'center', padding: 30 },
  iconWrapper: { backgroundColor: T.goldDim, padding: 20, borderRadius: 24, marginBottom: 20 },
  logo: { fontSize: 28, fontWeight: '900', color: T.text, letterSpacing: 2 },
  subtitle: { color: T.gold, fontSize: 14, fontWeight: '600', letterSpacing: 1, marginTop: 4 },
  inputContainer: { backgroundColor: T.surface, borderRadius: 16, marginBottom: 16, borderWidth: 1, borderColor: T.border, flexDirection: 'row', alignItems: 'center' },
  input: { flex: 1, color: T.text, padding: 18, fontSize: 16 },
  inputSuffix: { color: T.muted, paddingRight: 18, fontWeight: '600' },
  btnGold: { backgroundColor: T.gold, borderRadius: 16, padding: 18, alignItems: 'center', marginTop: 10, shadowColor: T.gold, shadowOpacity: 0.3, shadowRadius: 10, shadowOffset: {width:0, height:4}, elevation: 5 },
  btnGoldText: { color: T.bg, fontWeight: '800', fontSize: 16, letterSpacing: 1 },
  
  // Dashboard
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
  pageTitle: { fontSize: 28, fontWeight: '800', color: T.text, letterSpacing: -0.5 },
  statusBadge: (on) => ({ flexDirection: 'row', alignItems: 'center', backgroundColor: on ? '#00E67615' : '#FF3B3015', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, borderWidth: 1, borderColor: on ? '#00E67644' : '#FF3B3044' }),
  statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  statusBadgeText: { color: T.text, fontSize: 11, fontWeight: '800', letterSpacing: 1 },
  
  balanceCard: { padding: 24, borderRadius: 24, marginBottom: 20, borderWidth: 1, borderColor: T.goldDim },
  balanceLabel: { color: T.muted, fontSize: 13, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 1 },
  balanceValue: { color: T.text, fontSize: 42, fontWeight: '900', marginTop: 8, letterSpacing: -1 },
  
  statsRow: { flexDirection: 'row', marginBottom: 20 },
  statLabel: { color: T.muted, fontSize: 12, fontWeight: '600', textTransform: 'uppercase', marginBottom: 6 },
  statVal: { fontSize: 24, fontWeight: '800' },
  
  card: { backgroundColor: T.surface, borderRadius: 20, padding: 20, borderWidth: 1, borderColor: T.border },
  cardTitle: { color: T.text, fontSize: 15, fontWeight: '700' },
  
  row: { flexDirection: 'row', alignItems: 'center' },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  
  progressBar: { backgroundColor: T.bg, height: 8, borderRadius: 4, marginTop: 6, overflow: 'hidden' },
  progressFill: { backgroundColor: T.green, height: 8, borderRadius: 4 },
  
  actionBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 16, borderRadius: 16, marginRight: 8 },
  startBtn: { backgroundColor: T.gold },
  stopBtn: { backgroundColor: 'transparent', borderWidth: 2, borderColor: T.border, marginRight: 0 },
  actionBtnText: { color: T.bg, fontWeight: '800', fontSize: 14, letterSpacing: 1 },
  btnDisabled: { opacity: 0.4 },
  
  sectionTitle: { fontSize: 18, fontWeight: '700', color: T.text, marginBottom: 16 },
  posCard: { padding: 16, marginBottom: 12 },
  posIcon: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  posSymbol: { color: T.text, fontSize: 16, fontWeight: '800' },
  posVol: { fontSize: 12, fontWeight: '700', marginTop: 2 },
  posPnl: { fontSize: 16, fontWeight: '800' },
  posPrice: { color: T.muted, fontSize: 12, fontWeight: '600', marginTop: 2 },
  
  // Chips
  chip: { paddingHorizontal: 20, paddingVertical: 12, borderRadius: 100, backgroundColor: T.bg, borderWidth: 1, borderColor: T.border },
  chipActive: { backgroundColor: T.goldDim, borderColor: T.gold },
  chipText: { color: T.muted, fontSize: 13, fontWeight: '700' },
  chipTextActive: { color: T.gold },
  
  btnLogout: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 18, marginTop: 10, borderRadius: 16, backgroundColor: '#FF3B3015', borderWidth: 1, borderColor: '#FF3B3033' },
  btnLogoutText: { color: T.red, fontWeight: '700', marginLeft: 10 },
  
  muted: { color: T.muted, fontSize: 14 },
  mutedSmall: { color: T.muted, fontSize: 12, fontWeight: '600', textTransform: 'uppercase' },
});
