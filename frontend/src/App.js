import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  CircularProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [account, setAccount] = useState(null);
  const [positions, setPositions] = useState([]);
  const [symbol, setSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchAccountData();
    fetchPositions();
  }, []);

  const fetchAccountData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/account`);
      setAccount(response.data);
    } catch (err) {
      setError('Error fetching account data');
    }
  };

  const fetchPositions = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/positions`);
      setPositions(response.data);
    } catch (err) {
      setError('Error fetching positions');
    }
  };

  const handleTrade = async () => {
    if (!symbol) return;
    setLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/api/trade`, { symbol });
      fetchPositions();
      fetchAccountData();
    } catch (err) {
      setError('Error executing trade');
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          AI Trading Bot
        </Typography>

        <Grid container spacing={3}>
          {/* Account Overview */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Account Overview
                </Typography>
                {account ? (
                  <>
                    <Typography>Equity: ${account.equity.toFixed(2)}</Typography>
                    <Typography>Cash: ${account.cash.toFixed(2)}</Typography>
                    <Typography>
                      Buying Power: ${account.buying_power.toFixed(2)}
                    </Typography>
                  </>
                ) : (
                  <CircularProgress />
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Trading Interface */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Trading Interface
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <TextField
                    label="Symbol"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    fullWidth
                  />
                  <Button
                    variant="contained"
                    onClick={handleTrade}
                    disabled={loading || !symbol}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Trade'}
                  </Button>
                </Box>
                {error && (
                  <Typography color="error" gutterBottom>
                    {error}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Positions */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Current Positions
                </Typography>
                <Grid container spacing={2}>
                  {positions.map((position) => (
                    <Grid item xs={12} sm={6} md={4} key={position.symbol}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="h6">{position.symbol}</Typography>
                        <Typography>
                          Quantity: {position.qty}
                        </Typography>
                        <Typography>
                          Market Value: ${position.market_value.toFixed(2)}
                        </Typography>
                        <Typography
                          color={position.unrealized_pl >= 0 ? 'success.main' : 'error.main'}
                        >
                          Unrealized P/L: ${position.unrealized_pl.toFixed(2)}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default App; 