#ifndef CP_VERSION_H
#define CP_VERSION_H

/* The miner version: change it here only. Used by the pool agent string, the help text and
 * the stats API; scripts/release.py reads it for the release zip names. */
#define CP_VERSION "1.1"

/* Default --agent string sent to pools. */
#define CP_AGENT_DEFAULT "CPPminer/" CP_VERSION

#endif
