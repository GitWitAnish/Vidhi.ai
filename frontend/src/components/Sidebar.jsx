import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, PanelLeftClose, PanelLeft, LogOut } from "lucide-react";
import logo from "/logo.jpeg";
import "./Sidebar.css";

function Sidebar({ isOpen, onToggle, onNewChat }) {
  const sidebarWidth = window.innerWidth <= 480 ? 280 : 260;

  return (
    <>
      {/* Overlay for mobile */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="sidebar-overlay"
            onClick={onToggle}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          />
        )}
      </AnimatePresence>

      <motion.aside
        className="sidebar"
        initial={false}
        animate={{ x: isOpen ? 0 : -(sidebarWidth + 20) }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <motion.div
          className="sidebar-top"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <div className="sidebar-logo-section">
            <motion.img
              src={logo}
              alt="VIDHI.AI"
              className="sidebar-logo"
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
            />
            {/* <span className="sidebar-logo-text">VIDHI.AI</span> */}
          </div>
          <motion.button
            className="sidebar-close-btn"
            onClick={onToggle}
            aria-label="Close sidebar"
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
          >
            <PanelLeftClose size={20} />
          </motion.button>
        </motion.div>

        <motion.div
          className="sidebar-actions"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <motion.button
            className="new-chat-btn"
            onClick={onNewChat}
            whileHover={{ scale: 1.02, x: 5 }}
            whileTap={{ scale: 0.98 }}
            transition={{ duration: 0.2 }}
          >
            <motion.div
              whileHover={{ rotate: 90 }}
              transition={{ duration: 0.3 }}
            >
              <Plus size={20} />
            </motion.div>
            <span>New chat</span>
          </motion.button>
        </motion.div>

        <div className="sidebar-content">
          <div className="chat-history-section">
            <span className="history-label">Today</span>
            {/* Chat history items would go here */}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-divider" />
          <div className="user-section">
            <div className="user-avatar">A</div>
            <span className="user-name">Anish</span>
            <button className="logout-btn" aria-label="Logout">
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </motion.aside>
    </>
  );
}

export default Sidebar;
