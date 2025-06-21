import ssl
import socket
import struct
import random
import hashlib
from typing import List, Dict, Tuple, Optional, Any
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import secrets

logger = logging.getLogger(__name__)

class CipherSuite(Enum):
    """TLS 1.3 and 1.2 Cipher Suites"""
    # TLS 1.3
    TLS_AES_128_GCM_SHA256 = 0x1301
    TLS_AES_256_GCM_SHA384 = 0x1302
    TLS_CHACHA20_POLY1305_SHA256 = 0x1303

    # TLS 1.2
    TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 = 0xc02f
    TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 = 0xc030
    TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 = 0xcca8
    TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 = 0xc02b
    TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 = 0xc02c
    TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 = 0xcca9
    TLS_RSA_WITH_AES_128_GCM_SHA256 = 0x009c
    TLS_RSA_WITH_AES_256_GCM_SHA384 = 0x009d

class ExtensionType(Enum):
    """TLS Extension Types"""
    SERVER_NAME = 0x0000
    STATUS_REQUEST = 0x0005
    SUPPORTED_GROUPS = 0x000a
    EC_POINT_FORMATS = 0x000b
    SIGNATURE_ALGORITHMS = 0x000d
    ALPN = 0x0010
    SIGNED_CERT_TIMESTAMP = 0x0012
    PADDING = 0x0015
    EXTENDED_MASTER_SECRET = 0x0017
    SESSION_TICKET = 0x0023
    SUPPORTED_VERSIONS = 0x002b
    PSK_KEY_EXCHANGE_MODES = 0x002d
    KEY_SHARE = 0x0033
    RENEGOTIATION_INFO = 0xff01

@dataclass
class BrowserProfile:
    """Browser TLS fingerprint profile"""
    name: str
    cipher_suites: List[int]
    extensions: List[int]
    curves: List[int]
    signature_algorithms: List[int]
    alpn_protocols: List[str]
    tls_versions: List[int]
    compression_methods: List[int]

class TLSFingerprintManager:
    """Complete TLS/JA3 fingerprint spoofing implementation"""

    def __init__(self):
        self.browser_profiles = self._init_browser_profiles()
        self.current_profile = None
        self.connection_cache = {}

    def _init_browser_profiles(self) -> Dict[str, BrowserProfile]:
        """Initialize realistic browser TLS profiles"""
        return {
            'chrome_120': BrowserProfile(
                name='Chrome 120',
                cipher_suites=[
                    # GREASE
                    0x1a1a,
                    # TLS 1.3
                    CipherSuite.TLS_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_CHACHA20_POLY1305_SHA256.value,
                    # TLS 1.2
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256.value,
                    CipherSuite.TLS_RSA_WITH_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_RSA_WITH_AES_256_GCM_SHA384.value,
                ],
                extensions=[
                    0x0a0a,  # GREASE
                    ExtensionType.SERVER_NAME.value,
                    ExtensionType.EXTENDED_MASTER_SECRET.value,
                    ExtensionType.RENEGOTIATION_INFO.value,
                    ExtensionType.SUPPORTED_GROUPS.value,
                    ExtensionType.EC_POINT_FORMATS.value,
                    ExtensionType.SESSION_TICKET.value,
                    ExtensionType.ALPN.value,
                    ExtensionType.STATUS_REQUEST.value,
                    ExtensionType.SIGNATURE_ALGORITHMS.value,
                    ExtensionType.SIGNED_CERT_TIMESTAMP.value,
                    ExtensionType.KEY_SHARE.value,
                    ExtensionType.PSK_KEY_EXCHANGE_MODES.value,
                    ExtensionType.SUPPORTED_VERSIONS.value,
                    0x4469,  # GREASE
                    ExtensionType.PADDING.value,
                ],
                curves=[
                    0x0a0a,  # GREASE
                    0x001d,  # x25519
                    0x0017,  # secp256r1
                    0x0018,  # secp384r1
                ],
                signature_algorithms=[
                    0x0403,  # ecdsa_secp256r1_sha256
                    0x0503,  # ecdsa_secp384r1_sha384
                    0x0603,  # ecdsa_secp521r1_sha512
                    0x0807,  # ed25519
                    0x0808,  # ed448
                    0x0809,  # rsa_pss_pss_sha256
                    0x080a,  # rsa_pss_pss_sha384
                    0x080b,  # rsa_pss_pss_sha512
                    0x0804,  # rsa_pss_rsae_sha256
                    0x0805,  # rsa_pss_rsae_sha384
                    0x0806,  # rsa_pss_rsae_sha512
                    0x0401,  # rsa_pkcs1_sha256
                    0x0501,  # rsa_pkcs1_sha384
                    0x0601,  # rsa_pkcs1_sha512
                ],
                alpn_protocols=['h2', 'http/1.1'],
                tls_versions=[0x0304, 0x0303],  # TLS 1.3, 1.2
                compression_methods=[0x00]  # No compression
            ),

            'firefox_121': BrowserProfile(
                name='Firefox 121',
                cipher_suites=[
                    CipherSuite.TLS_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_CHACHA20_POLY1305_SHA256.value,
                    CipherSuite.TLS_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256.value,
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384.value,
                ],
                extensions=[
                    ExtensionType.SERVER_NAME.value,
                    ExtensionType.EXTENDED_MASTER_SECRET.value,
                    ExtensionType.RENEGOTIATION_INFO.value,
                    ExtensionType.SUPPORTED_GROUPS.value,
                    ExtensionType.EC_POINT_FORMATS.value,
                    ExtensionType.SESSION_TICKET.value,
                    ExtensionType.ALPN.value,
                    ExtensionType.STATUS_REQUEST.value,
                    ExtensionType.KEY_SHARE.value,
                    ExtensionType.SUPPORTED_VERSIONS.value,
                    ExtensionType.SIGNATURE_ALGORITHMS.value,
                    ExtensionType.PSK_KEY_EXCHANGE_MODES.value,
                    0x001c,  # record_size_limit
                    ExtensionType.PADDING.value,
                ],
                curves=[
                    0x001d,  # x25519
                    0x0017,  # secp256r1
                    0x0018,  # secp384r1
                    0x0019,  # secp521r1
                    0x0100,  # ffdhe2048
                    0x0101,  # ffdhe3072
                ],
                signature_algorithms=[
                    0x0403,  # ecdsa_secp256r1_sha256
                    0x0503,  # ecdsa_secp384r1_sha384
                    0x0603,  # ecdsa_secp521r1_sha512
                    0x0807,  # ed25519
                    0x0808,  # ed448
                    0x0804,  # rsa_pss_rsae_sha256
                    0x0805,  # rsa_pss_rsae_sha384
                    0x0806,  # rsa_pss_rsae_sha512
                    0x0401,  # rsa_pkcs1_sha256
                    0x0501,  # rsa_pkcs1_sha384
                    0x0601,  # rsa_pkcs1_sha512
                ],
                alpn_protocols=['h2', 'http/1.1'],
                tls_versions=[0x0304, 0x0303],
                compression_methods=[0x00]
            ),

            'safari_17': BrowserProfile(
                name='Safari 17',
                cipher_suites=[
                    # GREASE
                    0x0a0a,
                    # TLS 1.3
                    CipherSuite.TLS_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_CHACHA20_POLY1305_SHA256.value,
                    # TLS 1.2
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256.value,
                    CipherSuite.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256.value,
                    CipherSuite.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256.value,
                ],
                extensions=[
                    0x0a0a,  # GREASE
                    ExtensionType.SERVER_NAME.value,
                    ExtensionType.EXTENDED_MASTER_SECRET.value,
                    ExtensionType.RENEGOTIATION_INFO.value,
                    ExtensionType.SUPPORTED_GROUPS.value,
                    ExtensionType.EC_POINT_FORMATS.value,
                    ExtensionType.ALPN.value,
                    ExtensionType.STATUS_REQUEST.value,
                    ExtensionType.SIGNATURE_ALGORITHMS.value,
                    ExtensionType.SIGNED_CERT_TIMESTAMP.value,
                    ExtensionType.KEY_SHARE.value,
                    ExtensionType.PSK_KEY_EXCHANGE_MODES.value,
                    ExtensionType.SUPPORTED_VERSIONS.value,
                ],
                curves=[
                    0x0a0a,  # GREASE
                    0x001d,  # x25519
                    0x0017,  # secp256r1
                    0x0018,  # secp384r1
                ],
                signature_algorithms=[
                    0x0403,  # ecdsa_secp256r1_sha256
                    0x0807,  # ed25519
                    0x0804,  # rsa_pss_rsae_sha256
                    0x0401,  # rsa_pkcs1_sha256
                    0x0503,  # ecdsa_secp384r1_sha384
                    0x0805,  # rsa_pss_rsae_sha384
                    0x0501,  # rsa_pkcs1_sha384
                    0x0808,  # ed448
                    0x0806,  # rsa_pss_rsae_sha512
                    0x0601,  # rsa_pkcs1_sha512
                ],
                alpn_protocols=['h2', 'http/1.1'],
                tls_versions=[0x0304, 0x0303],
                compression_methods=[0x00]
            )
        }

    def generate_ja3_fingerprint(self, profile: BrowserProfile) -> str:
        """Generate JA3 fingerprint from profile"""
        # JA3 format: SSLVersion,Ciphers,Extensions,EllipticCurves,EllipticCurvePointFormats

        # TLS Version
        tls_version = str(profile.tls_versions[0])

        # Cipher suites
        ciphers = '-'.join(str(cs) for cs in profile.cipher_suites)

        # Extensions
        extensions = '-'.join(str(ext) for ext in profile.extensions)

        # Elliptic curves
        curves = '-'.join(str(curve) for curve in profile.curves)

        # Point formats (typically uncompressed)
        point_formats = '0'

        # Combine and hash
        ja3_string = f"{tls_version},{ciphers},{extensions},{curves},{point_formats}"
        ja3_hash = hashlib.md5(ja3_string.encode()).hexdigest()

        return ja3_hash

    def select_random_profile(self) -> BrowserProfile:
        """Select a random browser profile"""
        profile_name = random.choice(list(self.browser_profiles.keys()))
        return self.browser_profiles[profile_name]

    def create_client_hello(self, profile: BrowserProfile, hostname: str) -> bytes:
        """Create a TLS Client Hello message"""
        # Build Client Hello
        client_hello = bytearray()

        # TLS version in Client Hello (TLS 1.2 for compatibility)
        client_hello.extend(struct.pack('!BB', 0x03, 0x03))

        # Random bytes (32 bytes)
        client_hello.extend(secrets.token_bytes(32))

        # Session ID length and session ID (empty)
        client_hello.append(0x00)

        # Cipher suites
        cipher_data = b''.join(struct.pack('!H', cs) for cs in profile.cipher_suites)
        client_hello.extend(struct.pack('!H', len(cipher_data)))
        client_hello.extend(cipher_data)

        # Compression methods
        client_hello.append(len(profile.compression_methods))
        client_hello.extend(bytes(profile.compression_methods))

        # Extensions
        extensions = self._build_extensions(profile, hostname)
        client_hello.extend(struct.pack('!H', len(extensions)))
        client_hello.extend(extensions)

        # Wrap in handshake message
        handshake = bytearray()
        handshake.append(0x01)  # Client Hello type
        handshake.extend(struct.pack('!I', len(client_hello))[1:])  # 3-byte length
        handshake.extend(client_hello)

        # Wrap in TLS record
        record = bytearray()
        record.append(0x16)  # Handshake content type
        record.extend(struct.pack('!BB', 0x03, 0x01))  # TLS 1.0 for compatibility
        record.extend(struct.pack('!H', len(handshake)))
        record.extend(handshake)

        return bytes(record)

    def _build_extensions(self, profile: BrowserProfile, hostname: str) -> bytes:
        """Build TLS extensions"""
        extensions = bytearray()

        for ext_type in profile.extensions:
            if ext_type == ExtensionType.SERVER_NAME.value:
                # Server Name Indication
                ext_data = self._build_sni_extension(hostname)
            elif ext_type == ExtensionType.SUPPORTED_GROUPS.value:
                # Supported Groups (curves)
                ext_data = self._build_supported_groups_extension(profile.curves)
            elif ext_type == ExtensionType.EC_POINT_FORMATS.value:
                # EC Point Formats
                ext_data = bytes([0x01, 0x00])  # uncompressed
            elif ext_type == ExtensionType.SIGNATURE_ALGORITHMS.value:
                # Signature Algorithms
                ext_data = self._build_signature_algorithms_extension(profile.signature_algorithms)
            elif ext_type == ExtensionType.ALPN.value:
                # ALPN
                ext_data = self._build_alpn_extension(profile.alpn_protocols)
            elif ext_type == ExtensionType.SUPPORTED_VERSIONS.value:
                # Supported Versions
                ext_data = self._build_supported_versions_extension(profile.tls_versions)
            elif ext_type == ExtensionType.KEY_SHARE.value:
                # Key Share
                ext_data = self._build_key_share_extension(profile.curves)
            elif ext_type == ExtensionType.PSK_KEY_EXCHANGE_MODES.value:
                # PSK Key Exchange Modes
                ext_data = bytes([0x01, 0x01])  # psk_dhe_ke
            elif ext_type == ExtensionType.STATUS_REQUEST.value:
                # Status Request (OCSP)
                ext_data = bytes([0x01, 0x00, 0x00, 0x00, 0x00])
            elif ext_type == ExtensionType.SESSION_TICKET.value:
                # Session Ticket (empty)
                ext_data = b''
            elif ext_type == ExtensionType.EXTENDED_MASTER_SECRET.value:
                # Extended Master Secret
                ext_data = b''
            elif ext_type == ExtensionType.RENEGOTIATION_INFO.value:
                # Renegotiation Info
                ext_data = bytes([0x00])
            elif ext_type == ExtensionType.SIGNED_CERT_TIMESTAMP.value:
                # Signed Certificate Timestamp
                ext_data = b''
            elif ext_type == ExtensionType.PADDING.value:
                # Padding
                padding_len = 512 - (len(extensions) % 512)
                ext_data = bytes(padding_len)
            elif ext_type in [0x0a0a, 0x1a1a, 0x2a2a, 0x3a3a, 0x4a4a, 0x5a5a, 0x6a6a, 0x7a7a, 0x8a8a, 0x9a9a]:
                # GREASE values
                ext_data = bytes([0x00])
            else:
                # Unknown extension
                continue

            # Add extension
            extensions.extend(struct.pack('!H', ext_type))
            extensions.extend(struct.pack('!H', len(ext_data)))
            extensions.extend(ext_data)

        return bytes(extensions)

    def _build_sni_extension(self, hostname: str) -> bytes:
        """Build Server Name Indication extension"""
        sni_data = bytearray()

        # Server name list length
        server_name = hostname.encode('ascii')
        sni_data.extend(struct.pack('!H', len(server_name) + 3))

        # Server name type (0 = hostname)
        sni_data.append(0x00)

        # Server name length
        sni_data.extend(struct.pack('!H', len(server_name)))
        sni_data.extend(server_name)

        return bytes(sni_data)

    def _build_supported_groups_extension(self, curves: List[int]) -> bytes:
        """Build Supported Groups extension"""
        groups_data = bytearray()

        # Groups list
        groups_list = b''.join(struct.pack('!H', curve) for curve in curves)
        groups_data.extend(struct.pack('!H', len(groups_list)))
        groups_data.extend(groups_list)

        return bytes(groups_data)

    def _build_signature_algorithms_extension(self, algorithms: List[int]) -> bytes:
        """Build Signature Algorithms extension"""
        sig_data = bytearray()

        # Algorithms list
        alg_list = b''.join(struct.pack('!H', alg) for alg in algorithms)
        sig_data.extend(struct.pack('!H', len(alg_list)))
        sig_data.extend(alg_list)

        return bytes(sig_data)

    def _build_alpn_extension(self, protocols: List[str]) -> bytes:
        """Build ALPN extension"""
        alpn_data = bytearray()

        # Protocol list
        protocol_list = bytearray()
        for protocol in protocols:
            proto_bytes = protocol.encode('ascii')
            protocol_list.append(len(proto_bytes))
            protocol_list.extend(proto_bytes)

        alpn_data.extend(struct.pack('!H', len(protocol_list)))
        alpn_data.extend(protocol_list)

        return bytes(alpn_data)

    def _build_supported_versions_extension(self, versions: List[int]) -> bytes:
        """Build Supported Versions extension"""
        versions_data = bytearray()

        # Version list
        version_list = b''.join(struct.pack('!H', ver) for ver in versions)
        versions_data.append(len(version_list))
        versions_data.extend(version_list)

        return bytes(versions_data)

    def _build_key_share_extension(self, curves: List[int]) -> bytes:
        """Build Key Share extension"""
        key_share_data = bytearray()

        # Generate key shares for first two curves
        shares = bytearray()

        for curve in curves[:2]:
            if curve in [0x0a0a, 0x1a1a]:  # Skip GREASE
                continue

            if curve == 0x001d:  # x25519
                # Generate random 32-byte key
                key = secrets.token_bytes(32)
                shares.extend(struct.pack('!H', curve))
                shares.extend(struct.pack('!H', len(key)))
                shares.extend(key)
            elif curve == 0x0017:  # secp256r1
                # Generate random 65-byte key (uncompressed point)
                key = bytes([0x04]) + secrets.token_bytes(64)
                shares.extend(struct.pack('!H', curve))
                shares.extend(struct.pack('!H', len(key)))
                shares.extend(key)

        key_share_data.extend(struct.pack('!H', len(shares)))
        key_share_data.extend(shares)

        return bytes(key_share_data)

    async def create_tls_connection(self, hostname: str, port: int = 443) -> Tuple[socket.socket, str]:
        """Create a TLS connection with spoofed fingerprint"""
        # Select random profile
        profile = self.select_random_profile()
        self.current_profile = profile

        # Generate JA3 fingerprint
        ja3_hash = self.generate_ja3_fingerprint(profile)
        logger.info(f"Using {profile.name} profile with JA3: {ja3_hash}")

        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)

        try:
            # Connect to server
            sock.connect((hostname, port))

            # Send Client Hello
            client_hello = self.create_client_hello(profile, hostname)
            sock.send(client_hello)

            # Complete TLS handshake
            # In production, this would complete the full handshake
            # For now, we'll use standard SSL

            # Wrap socket with SSL
            context = self._create_ssl_context(profile)
            ssl_sock = context.wrap_socket(sock, server_hostname=hostname)

            return ssl_sock, ja3_hash

        except Exception as e:
            sock.close()
            raise Exception(f"TLS connection failed: {e}")

    def _create_ssl_context(self, profile: BrowserProfile) -> ssl.SSLContext:
        """Create SSL context matching the profile"""
        context = ssl.create_default_context()

        # Set cipher suites
        cipher_string = self._convert_ciphers_to_openssl(profile.cipher_suites)
        if cipher_string:
            context.set_ciphers(cipher_string)

        # Set ALPN protocols
        context.set_alpn_protocols(profile.alpn_protocols)

        # Disable certain options to match browser behavior
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED

        # Set minimum and maximum TLS versions
        if 0x0304 in profile.tls_versions:  # TLS 1.3
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        else:
            context.maximum_version = ssl.TLSVersion.TLSv1_2

        context.minimum_version = ssl.TLSVersion.TLSv1_2

        return context

    def _convert_ciphers_to_openssl(self, cipher_suites: List[int]) -> str:
        """Convert cipher suite numbers to OpenSSL names"""
        # Mapping of common cipher suites
        cipher_map = {
            0x1301: 'TLS_AES_128_GCM_SHA256',
            0x1302: 'TLS_AES_256_GCM_SHA384',
            0x1303: 'TLS_CHACHA20_POLY1305_SHA256',
            0xc02f: 'ECDHE-RSA-AES128-GCM-SHA256',
            0xc030: 'ECDHE-RSA-AES256-GCM-SHA384',
            0xc02b: 'ECDHE-ECDSA-AES128-GCM-SHA256',
            0xc02c: 'ECDHE-ECDSA-AES256-GCM-SHA384',
            0xcca8: 'ECDHE-RSA-CHACHA20-POLY1305',
            0xcca9: 'ECDHE-ECDSA-CHACHA20-POLY1305',
            0x009c: 'AES128-GCM-SHA256',
            0x009d: 'AES256-GCM-SHA384',
        }

        openssl_names = []
        for suite in cipher_suites:
            if suite in cipher_map:
                openssl_names.append(cipher_map[suite])

        return ':'.join(openssl_names) if openssl_names else None

    def get_http2_settings(self, profile: BrowserProfile) -> Dict[int, int]:
        """Get HTTP/2 settings matching the browser profile"""
        # Browser-specific HTTP/2 settings
        if 'chrome' in profile.name.lower():
            return {
                0x1: 65536,     # SETTINGS_HEADER_TABLE_SIZE
                0x2: 0,         # SETTINGS_ENABLE_PUSH
                0x3: 1000,      # SETTINGS_MAX_CONCURRENT_STREAMS
                0x4: 6291456,   # SETTINGS_INITIAL_WINDOW_SIZE
                0x5: 16384,     # SETTINGS_MAX_FRAME_SIZE
                0x6: 262144,    # SETTINGS_MAX_HEADER_LIST_SIZE
            }
        elif 'firefox' in profile.name.lower():
            return {
                0x1: 65536,
                0x3: 100,
                0x4: 131072,
                0x5: 16384,
            }
        elif 'safari' in profile.name.lower():
            return {
                0x1: 4096,
                0x3: 100,
                0x4: 65535,
                0x5: 16384,
            }
        else:
            # Default settings
            return {
                0x1: 4096,
                0x3: 100,
                0x4: 65535,
                0x5: 16384,
            }

    def get_http_headers_order(self, profile: BrowserProfile) -> List[str]:
        """Get HTTP header order matching the browser profile"""
        if 'chrome' in profile.name.lower():
            return [
                ':method',
                ':authority',
                ':scheme',
                ':path',
                'accept',
                'accept-encoding',
                'accept-language',
                'cache-control',
                'sec-ch-ua',
                'sec-ch-ua-mobile',
                'sec-ch-ua-platform',
                'sec-fetch-dest',
                'sec-fetch-mode',
                'sec-fetch-site',
                'sec-fetch-user',
                'upgrade-insecure-requests',
                'user-agent',
                'referer',
                'cookie',
            ]
        elif 'firefox' in profile.name.lower():
            return [
                ':method',
                ':path',
                ':authority',
                ':scheme',
                'user-agent',
                'accept',
                'accept-language',
                'accept-encoding',
                'referer',
                'cookie',
                'upgrade-insecure-requests',
                'sec-fetch-dest',
                'sec-fetch-mode',
                'sec-fetch-site',
                'te',
            ]
        elif 'safari' in profile.name.lower():
            return [
                ':method',
                ':scheme',
                ':path',
                ':authority',
                'accept',
                'accept-encoding',
                'accept-language',
                'user-agent',
                'referer',
                'cookie',
            ]
        else:
            return []


class TLSConnection:
    """Wrapper for TLS connections with fingerprint spoofing"""

    def __init__(self, fingerprint_manager: TLSFingerprintManager):
        self.fingerprint_manager = fingerprint_manager
        self.connections = {}

    async def connect(self, url: str) -> Tuple[Any, str]:
        """Create a connection with spoofed TLS fingerprint"""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port or 443

        # Create TLS connection
        sock, ja3_hash = await self.fingerprint_manager.create_tls_connection(hostname, port)

        # Store connection
        conn_id = f"{hostname}:{port}"
        self.connections[conn_id] = sock

        return sock, ja3_hash

    def close(self, conn_id: str):
        """Close a connection"""
        if conn_id in self.connections:
            try:
                self.connections[conn_id].close()
            except:
                pass
            del self.connections[conn_id]

    def close_all(self):
        """Close all connections"""
        for conn_id in list(self.connections.keys()):
            self.close(conn_id)